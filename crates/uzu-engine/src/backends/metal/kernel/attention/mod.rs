mod policy;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding,
            kernel::{AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, kv_layout::KVLayout},
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            gpu_types::{attention::AttnParams, kv_layout},
            kernel::{AttentionMatrixCombineMetalKernel, AttentionMatrixMetalKernel, AttentionVectorMetalKernel},
        },
    },
    data_type::DataType,
};

pub struct AttentionMetalKernel {
    config: AttentionKernelConfig,
    use_mxu: bool,
    tiles: Vec<KernelTile>,
    vector: Vec<[AttentionVectorMetalKernel; 2]>,
    combine: AttentionMatrixCombineMetalKernel,
}

struct KernelTile {
    rows: u32,
    simdgroups: u32,
    prefix: Option<AttentionMatrixMetalKernel>,
    masked: AttentionMatrixMetalKernel,
    trie: AttentionMatrixMetalKernel,
}

impl AttentionMetalKernel {
    fn new_with_mxu(
        context: &MetalContext,
        config: &AttentionKernelConfig,
        use_mxu: bool,
    ) -> Result<Self, MetalError> {
        let slices = (config.head_dim
            / if use_mxu {
                kv_layout::SLICE_DIMS
            } else {
                128
            })
        .max(1);
        let mut tiles = Vec::new();
        for simdgroups in [1, 2, 4, 6, 8] {
            if simdgroups % slices != 0 {
                continue;
            }
            let kernel = |mask_free, is_trie| {
                AttentionMatrixMetalKernel::new(
                    context,
                    DataType::BF16,
                    config.head_dim,
                    use_mxu,
                    simdgroups,
                    config.is_causal,
                    config.is_kv_cache_ring,
                    config.sliding_window_size.is_some(),
                    mask_free,
                    is_trie,
                    config.num_groups,
                )
            };
            tiles.push(KernelTile {
                simdgroups,
                rows: if use_mxu {
                    simdgroups * kv_layout::TILE_QUERIES / (config.head_dim / kv_layout::SLICE_DIMS)
                } else {
                    simdgroups * 8 / slices
                },
                prefix: (use_mxu && !config.is_kv_cache_ring && config.sliding_window_size.is_none())
                    .then(|| kernel(true, false))
                    .transpose()?,
                masked: kernel(false, false)?,
                trie: kernel(false, true)?,
            });
        }
        let mut vector = Vec::new();
        if !use_mxu {
            for (queries, split_kv) in [(1, false), (1, true), (2, true)] {
                let kernel = |is_trie| {
                    AttentionVectorMetalKernel::new(
                        context,
                        if split_kv {
                            DataType::BF16
                        } else {
                            config.data_type
                        },
                        config.head_dim,
                        queries,
                        split_kv,
                        config.is_causal,
                        config.is_kv_cache_ring,
                        config.sliding_window_size.is_some(),
                        is_trie,
                        !split_kv && config.gate_row_stride.is_some(),
                        !split_kv && config.has_sinks,
                        config.num_groups,
                    )
                };
                vector.push([kernel(false)?, kernel(true)?]);
            }
        }
        Ok(Self {
            config: *config,
            use_mxu,
            tiles,
            vector,
            combine: AttentionMatrixCombineMetalKernel::new(
                context,
                config.data_type,
                config.head_dim,
                config.gate_row_stride.is_some(),
                config.has_sinks,
            )?,
        })
    }
}

impl AttentionKernel for AttentionMetalKernel {
    type Backend = Metal;
    type Error = MetalError;

    fn kv_layout(&self) -> KVLayout {
        let layout = if self.use_mxu {
            kv_layout::KVLayout::MxuTiled
        } else {
            kv_layout::KVLayout::RowMajor
        };
        KVLayout::new(self.config.head_dim, self.config.num_groups, kv_layout::TILE_KEYS, layout as u32)
    }

    fn new(
        context: &MetalContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, MetalError> {
        Self::new_with_mxu(context, &config, context.supports_mxu && config.data_type == DataType::BF16)
    }

    fn encode(
        &self,
        input: AttentionInput<
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        arguments: AttentionArguments<'_, Metal, impl BufferRef<Backend = Metal>, impl BufferRef<Backend = Metal>>,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let command_buffer = &mut *command_buffer.debug_group("attention core");
        let AttentionArguments {
            suffix_length,
            trie,
            cache,
            gate,
            sinks,
        } = arguments;
        let config = &self.config;
        let (head_dim, num_q_heads, num_groups) = (config.head_dim, config.num_q_heads, config.num_groups);
        let prefix_len = cache.prefix_len();
        let kv_length = prefix_len + suffix_length;

        let gqa_factor = num_q_heads / num_groups;
        let grouped_rows = gqa_factor * suffix_length;
        let params = AttnParams {
            q_strides: [0, grouped_rows * head_dim, head_dim],
            k_strides: [0, head_dim, num_groups * head_dim],
            v_strides: [0, head_dim, num_groups * head_dim],
            o_strides: [0, head_dim, num_q_heads * head_dim],
            gqa_factor,
            scale: config.scale.unwrap_or(1.0 / (head_dim as f32).sqrt()),
            q_len: suffix_length,
            k_len: kv_length,
            q_off: prefix_len,
            nk: kv_length.div_ceil(kv_layout::TILE_KEYS),
        };
        let direct = !self.use_mxu && kv_length <= kv_layout::TILE_KEYS;
        let vector_queries = if grouped_rows.is_multiple_of(2) {
            2
        } else {
            1
        };
        let tile = (self.use_mxu || (!direct && grouped_rows > 4)).then(|| {
            self.tiles
                .iter()
                .min_by_key(|tile| (grouped_rows.div_ceil(tile.rows) * tile.rows, std::cmp::Reverse(tile.rows)))
                .unwrap()
        });
        let padded_rows = tile.map_or(grouped_rows, |tile| grouped_rows.next_multiple_of(tile.rows));
        let split_rows = num_groups * padded_rows;
        let num_splits = if direct {
            1
        } else {
            let (threadgroups, groups_per_core) = match tile {
                None => (split_rows.div_ceil(4 * vector_queries), 4),
                Some(tile) => (num_groups * (padded_rows / tile.rows), 8_u32.div_ceil(tile.simdgroups).max(2)),
            };
            let occupancy = (!self.use_mxu).then(|| {
                (command_buffer.context().gpu_core_count.max(1) * groups_per_core).div_ceil(threadgroups.max(1))
            });
            policy::splits(&params, head_dim, split_rows, occupancy)
        };

        let rows = split_rows * num_splits;
        let data_type = if direct {
            config.data_type
        } else {
            DataType::BF16
        };
        let mut values = command_buffer.allocate_scratch_for_shape(&[rows, head_dim], data_type)?;
        let mut statistics = if direct {
            None
        } else {
            Some((
                command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?,
                command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?,
            ))
        };
        let (maxima, sums) = statistics.as_mut().map_or((None, None), |(maxima, sums)| (Some(maxima), Some(sums)));
        if let Some(tile) = tile {
            let query_tiles = padded_rows / tile.rows;
            let prefix_splits = tile.prefix.as_ref().map_or(0, |_| {
                (0..num_splits)
                    .take_while(|split| ((split + 1) * params.nk / num_splits) * kv_layout::TILE_KEYS <= params.q_off)
                    .count() as u32
            });
            let maxima = maxima.unwrap();
            let sums = sums.unwrap();
            for (kernel, trie, split_base, split_count) in [
                (tile.prefix.as_ref().unwrap_or(&tile.masked), None, 0, prefix_splits),
                (trie.map_or(&tile.masked, |_| &tile.trie), trie, prefix_splits, num_splits - prefix_splits),
            ] {
                if split_count == 0 {
                    continue;
                }
                let range = |row_bytes: usize| {
                    let bytes = split_rows as usize * row_bytes;
                    split_base as usize * bytes..(split_base + split_count) as usize * bytes
                };
                kernel.encode(
                    input.query_codes,
                    input.query_scales,
                    input.k_codes,
                    input.k_scales,
                    input.v_codes,
                    input.v_scales,
                    trie,
                    cache.ring_params(),
                    config.sliding_window_size,
                    (&mut values).subrange_mut(range(head_dim as usize * DataType::BF16.size_in_bytes())),
                    (&mut *maxima).subrange_mut(range(DataType::F32.size_in_bytes())),
                    (&mut *sums).subrange_mut(range(DataType::F32.size_in_bytes())),
                    params,
                    query_tiles,
                    num_splits,
                    split_base,
                    split_count,
                    suffix_length,
                    command_buffer,
                );
            }
        } else {
            let variant = if direct {
                0
            } else {
                vector_queries as usize
            };
            self.vector[variant][usize::from(trie.is_some())].encode(
                input.query_codes,
                input.query_scales,
                input.k_codes,
                input.k_scales,
                input.v_codes,
                input.v_scales,
                &mut values,
                maxima,
                sums,
                trie,
                cache.ring_params(),
                config.sliding_window_size,
                gate.filter(|_| direct),
                sinks.filter(|_| direct),
                config.gate_row_stride.filter(|_| direct),
                params,
                split_rows,
                num_splits,
                command_buffer,
            );
        }

        let Some((maxima, sums)) = statistics else {
            return Ok(values);
        };
        let mut output =
            command_buffer.allocate_scratch_for_shape(&[suffix_length, num_q_heads, head_dim], config.data_type)?;
        self.combine.encode(
            &values,
            &maxima,
            &sums,
            &mut output,
            gate,
            sinks,
            params,
            grouped_rows,
            padded_rows,
            num_groups,
            num_splits,
            config.gate_row_stride,
            command_buffer,
        );
        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/attention/mod.rs"]
mod tests;

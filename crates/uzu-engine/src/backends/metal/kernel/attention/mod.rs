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
            kernel::{
                AttentionBlockScoreMetalKernel, AttentionMatrixCombineMetalKernel, AttentionMatrixMetalKernel,
                AttentionVectorMetalKernel,
            },
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
    sparse: Option<(u32, u32, [AttentionMatrixMetalKernel; 2])>,
}

struct KernelTile {
    rows: u32,
    simdgroups: u32,
    mask_free_split: Option<AttentionMatrixMetalKernel>,
    masked_split: AttentionMatrixMetalKernel,
    trie_split: AttentionMatrixMetalKernel,
    masked: Option<AttentionMatrixMetalKernel>,
    trie: Option<AttentionMatrixMetalKernel>,
}

impl KernelTile {
    fn output(
        &self,
        is_trie: bool,
    ) -> Option<&AttentionMatrixMetalKernel> {
        if is_trie {
            self.trie.as_ref()
        } else {
            self.masked.as_ref()
        }
    }

    fn split(
        &self,
        is_trie: bool,
    ) -> &AttentionMatrixMetalKernel {
        if is_trie {
            &self.trie_split
        } else {
            &self.masked_split
        }
    }
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
        let tile_rows = |simdgroups| {
            simdgroups
                * if use_mxu {
                    kv_layout::TILE_QUERIES
                } else {
                    8
                }
                / slices
        };
        let mut tiles = Vec::new();
        for simdgroups in [1, 2, 4, 6, 8] {
            if simdgroups % slices != 0 {
                continue;
            }
            let kernel = |mask_free, is_trie, is_split: bool| {
                AttentionMatrixMetalKernel::new(
                    context,
                    DataType::BF16,
                    config.head_dim,
                    use_mxu,
                    simdgroups,
                    false,
                    config.is_causal,
                    config.is_kv_cache_ring,
                    config.sliding_window_size.is_some(),
                    mask_free,
                    is_trie,
                    is_split,
                    !is_split && config.gate_row_stride.is_some(),
                    !is_split && config.has_sinks,
                    config.num_groups,
                    None,
                )
            };
            // TODO: simdgroup (pre-M5) kernels have no no-split path yet, so without MXU every call writes partials.
            let output = |is_trie| use_mxu.then(|| kernel(false, is_trie, false)).transpose();
            let plain_causal = use_mxu && !config.is_kv_cache_ring && config.sliding_window_size.is_none();
            tiles.push(KernelTile {
                simdgroups,
                rows: tile_rows(simdgroups),
                masked: output(false)?,
                trie: output(true)?,
                masked_split: kernel(false, false, true)?,
                trie_split: kernel(false, true, true)?,
                mask_free_split: plain_causal.then(|| kernel(true, false, true)).transpose()?,
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
            sparse: config
                .sparse_compression_ratio
                .map(|compression_ratio| {
                    if config.is_kv_cache_ring || config.sliding_window_size.is_some() || !config.is_causal {
                        return Err(MetalError::CannotCreateFunction(
                            "sparse attention is causal without window or ring".to_string(),
                        ));
                    }
                    if compression_ratio % kv_layout::LANE_KEYS != 0 {
                        return Err(MetalError::CannotCreateFunction(format!(
                            "sparse blocks of {compression_ratio} keys must fill whole lane groups of {} keys",
                            kv_layout::LANE_KEYS
                        )));
                    }
                    let simdgroups = config.head_dim / kv_layout::SLICE_DIMS;
                    let rows = tile_rows(simdgroups);
                    if config.num_q_heads / config.num_groups > rows {
                        return Err(MetalError::CannotCreateFunction(format!(
                            "sparse attention covers at most {rows} GQA rows"
                        )));
                    }
                    let kernel = |is_trie| {
                        AttentionMatrixMetalKernel::new(
                            context,
                            DataType::BF16,
                            config.head_dim,
                            use_mxu,
                            simdgroups,
                            true,
                            false,
                            false,
                            false,
                            false,
                            is_trie,
                            true,
                            false,
                            false,
                            config.num_groups,
                            Some(compression_ratio),
                        )
                    };
                    Ok((rows, compression_ratio, [kernel(false)?, kernel(true)?]))
                })
                .transpose()?,
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
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let command_buffer = &mut *command_buffer.debug_group("attention core");
        let AttentionArguments {
            suffix_length,
            trie,
            key_blocks,
            cache,
            gate,
            sinks,
        } = arguments;
        let config = &self.config;
        let (head_dim, num_q_heads, num_groups) = (config.head_dim, config.num_q_heads, config.num_groups);
        let prefix_len = cache.prefix_len();
        let kv_length = prefix_len + suffix_length;

        assert!(
            cache
                .ring_params()
                .is_none_or(|ring| ring.ring_length == prefix_len && ring.ring_offset < prefix_len.max(1)),
            "attention assumes a full ring"
        );
        let gqa_factor = num_q_heads / num_groups;
        assert!(
            !self.use_mxu || input.k_codes.size().max(input.v_codes.size()) as u64 <= u64::from(u32::MAX),
            "the MXU core addresses its K and V cache with 32-bit byte offsets"
        );
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
        let sparse = match (key_blocks, &self.sparse) {
            (None, None) => None,
            (Some(key_blocks), Some((rows, compression_ratio, cores))) => {
                Some((key_blocks, &cores[usize::from(trie.is_some())], *rows, *compression_ratio))
            },
            _ => panic!("key_blocks must match AttentionKernelConfig::sparse_compression_ratio"),
        };
        let direct = sparse.is_none() && !self.use_mxu && kv_length <= kv_layout::TILE_KEYS;
        let vector_queries = if grouped_rows.is_multiple_of(2) {
            2
        } else {
            1
        };
        let tile = (self.use_mxu || (!direct && grouped_rows > 4)).then(|| {
            self.tiles
                .iter()
                .min_by_key(|tile| policy::tile_order(tile.rows, grouped_rows, suffix_length, self.use_mxu))
                .unwrap()
        });
        // Sparse: every query is padded to one tile of rows.
        let padded_rows = if let Some((.., rows, _)) = &sparse {
            rows * suffix_length
        } else {
            tile.map_or(grouped_rows, |tile| grouped_rows.next_multiple_of(tile.rows))
        };
        let split_rows = num_groups * padded_rows;
        let num_splits = if direct {
            1
        } else if let Some((key_blocks, .., compression_ratio)) = &sparse {
            policy::sparse_splits(
                command_buffer.context().gpu_core_count.max(1),
                suffix_length * num_groups,
                key_blocks.blocks_per_query,
                *compression_ratio,
            )
        } else {
            let threadgroups =
                tile.map_or(split_rows.div_ceil(4 * vector_queries), |tile| num_groups * (padded_rows / tile.rows));
            policy::splits(
                &params,
                head_dim,
                split_rows,
                command_buffer.context().gpu_core_count.max(1),
                threadgroups,
                tile.map(|tile| tile.simdgroups),
                self.use_mxu,
            )
        };

        let output_kernel = tile.and_then(|tile| tile.output(trie.is_some()));
        let is_split = !direct && (sparse.is_some() || num_splits > 1 || output_kernel.is_none());
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
        let mut output = (!direct)
            .then(|| {
                command_buffer.allocate_scratch_for_shape(&[suffix_length, num_q_heads, head_dim], config.data_type)
            })
            .transpose()?;
        if let Some((key_blocks, core, ..)) = sparse {
            let (maxima, sums) = statistics.as_mut().unwrap();
            core.encode(
                input.query_codes,
                input.query_scales,
                input.k_codes,
                input.k_scales,
                input.v_codes,
                input.v_scales,
                trie,
                cache.ring_params(),
                config.sliding_window_size,
                Some(&mut values),
                Some(maxima),
                Some(sums),
                None::<&mut <Metal as Backend>::ScratchBuffer>,
                None::<&<Metal as Backend>::GlobalGpuBuffer>,
                None,
                None::<&<Metal as Backend>::GlobalGpuBuffer>,
                params,
                suffix_length,
                num_splits,
                0,
                num_splits,
                suffix_length,
                Some(key_blocks.query_blocks),
                Some(key_blocks.blocks_per_query),
                command_buffer,
            );
        } else if let Some(tile) = tile {
            let query_tiles = padded_rows / tile.rows;
            let prefix = tile.mask_free_split.as_ref().filter(|_| is_split);
            let prefix_splits = prefix.map_or(0, |_| {
                (0..num_splits)
                    .take_while(|split| ((split + 1) * params.nk / num_splits) * kv_layout::TILE_KEYS <= params.q_off)
                    .count() as u32
            });
            for (kernel, trie, split_base, split_count) in [
                (prefix.unwrap_or(&tile.masked_split), None, 0, prefix_splits),
                (
                    output_kernel.filter(|_| !is_split).unwrap_or(tile.split(trie.is_some())),
                    trie,
                    prefix_splits,
                    num_splits - prefix_splits,
                ),
            ] {
                if split_count == 0 {
                    continue;
                }
                let range = |row_bytes: usize| {
                    let bytes = split_rows as usize * row_bytes;
                    split_base as usize * bytes..(split_base + split_count) as usize * bytes
                };
                let (maxima, sums) = statistics.as_mut().unwrap();
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
                    is_split.then_some(
                        (&mut values).subrange_mut(range(head_dim as usize * DataType::BF16.size_in_bytes())),
                    ),
                    is_split.then_some(maxima.subrange_mut(range(DataType::F32.size_in_bytes()))),
                    is_split.then_some(sums.subrange_mut(range(DataType::F32.size_in_bytes()))),
                    output.as_mut().filter(|_| !is_split),
                    gate.filter(|_| !is_split),
                    config.gate_row_stride.filter(|_| !is_split),
                    sinks.filter(|_| !is_split),
                    params,
                    query_tiles,
                    num_splits,
                    split_base,
                    split_count,
                    suffix_length,
                    None::<&<Metal as Backend>::ConstantBuffer>,
                    None,
                    command_buffer,
                );
            }
        } else {
            let variant = if direct {
                0
            } else {
                vector_queries as usize
            };
            let (maxima, sums) = statistics.as_mut().map_or((None, None), |(maxima, sums)| (Some(maxima), Some(sums)));
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

        let Some((maxima, sums)) = statistics.filter(|_| is_split) else {
            return Ok(output.unwrap_or(values));
        };
        let mut output = output.unwrap();
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

// Keeps AttentionBlockScore compiled until the QSA indexer wiring calls it.
#[allow(dead_code)]
fn encode_block_score(
    context: &MetalContext,
    buffer: impl BufferRef<Backend = Metal>,
    scores: impl BufferMut<Backend = Metal>,
    encoder: &mut MetalCommandBufferEncoding,
) -> Result<(), MetalError> {
    AttentionBlockScoreMetalKernel::new(context, 4, 128, false, 4, false)?.encode(
        buffer,
        buffer,
        buffer,
        buffer,
        scores,
        0,
        0,
        0,
        0,
        0,
        16,
        32,
        0,
        None::<&<Metal as Backend>::GlobalGpuBuffer>,
        encoder,
    );
    Ok(())
}

mod error;
mod policy;

pub use error::AttentionMetalError;
use policy::{MAX_TRIE_SUFFIX, int8_simd_splits, int8_splits};

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding,
            gpu_types::kv_layout,
            kernel::{
                AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, kv_layout::Int8KVLayout,
            },
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            gpu_types::attention::AttnParams,
            kernel::{
                AttentionGemmGroupedCombineMetalKernel, AttentionGemmGroupedMetalKernel, AttentionSmallMetalKernel,
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
    vector: Option<VectorKernels>,
    combine: AttentionGemmGroupedCombineMetalKernel,
}

struct VectorKernels {
    direct: [AttentionSmallMetalKernel; 2],
    split: [[AttentionVectorMetalKernel; 2]; 2],
}

struct KernelTile {
    rows: u32,
    simdgroups: u32,
    prefix: Option<AttentionGemmGroupedMetalKernel>,
    masked: AttentionGemmGroupedMetalKernel,
    trie: AttentionGemmGroupedMetalKernel,
}

impl AttentionMetalKernel {
    fn is_supported(config: &AttentionKernelConfig) -> bool {
        matches!(config.head_dim, 64 | 128 | 256 | 512)
            && matches!(config.data_type, DataType::BF16 | DataType::F32)
            && config.num_groups > 0
            && config.num_q_heads > 0
            && config.num_q_heads.is_multiple_of(config.num_groups)
    }

    fn new_with_mxu(
        context: &MetalContext,
        config: &AttentionKernelConfig,
        use_mxu: bool,
    ) -> Result<Self, MetalError> {
        let slices = (config.head_dim / 128).max(1);
        let mut tiles = Vec::new();
        for simdgroups in [1, 2, 4, 6, 8] {
            if (use_mxu && simdgroups != 8) || (!use_mxu && simdgroups % slices != 0) {
                continue;
            }
            let kernel = |mask_free, is_trie| {
                AttentionGemmGroupedMetalKernel::new(
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
                prefix: use_mxu.then(|| kernel(true, false)).transpose()?,
                masked: kernel(false, false)?,
                trie: kernel(false, true)?,
            });
        }
        let vector_kernel = |queries, is_trie| {
            AttentionVectorMetalKernel::new(
                context,
                config.head_dim,
                queries,
                config.is_causal,
                config.is_kv_cache_ring,
                config.sliding_window_size.is_some(),
                is_trie,
                config.num_groups,
            )
        };
        let small_kernel = |is_trie| {
            AttentionSmallMetalKernel::new(
                context,
                config.data_type,
                config.head_dim,
                config.is_causal,
                config.is_kv_cache_ring,
                config.sliding_window_size.is_some(),
                is_trie,
                config.gate_row_stride.is_some(),
                config.has_sinks,
                config.num_groups,
            )
        };
        Ok(Self {
            config: *config,
            use_mxu,
            tiles,
            vector: if use_mxu {
                None
            } else {
                Some(VectorKernels {
                    direct: [small_kernel(false)?, small_kernel(true)?],
                    split: [
                        [vector_kernel(1, false)?, vector_kernel(1, true)?],
                        [vector_kernel(2, false)?, vector_kernel(2, true)?],
                    ],
                })
            },
            combine: AttentionGemmGroupedCombineMetalKernel::new(
                context,
                config.data_type,
                config.head_dim,
                config.gate_row_stride.is_some(),
                config.has_sinks,
            )?,
        })
    }

    fn encode_attention(
        &self,
        input: AttentionInput<
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        arguments: AttentionArguments<'_, Metal, impl BufferRef<Backend = Metal>, impl BufferRef<Backend = Metal>>,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        assert!(
            !self.use_mxu || arguments.trie.is_none() || arguments.suffix_length <= MAX_TRIE_SUFFIX,
            "trie suffix is not supported"
        );
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
        if let Some(vector) = &self.vector
            && kv_length <= kv_layout::TILE_KEYS
        {
            let mut output =
                command_buffer.allocate_scratch_for_shape(&[suffix_length, num_q_heads, head_dim], config.data_type)?;
            vector.direct[usize::from(trie.is_some())].encode(
                input.query_codes,
                input.query_scales,
                input.k_codes,
                input.k_scales,
                input.v_codes,
                input.v_scales,
                &mut output,
                trie,
                cache.ring_params(),
                config.sliding_window_size,
                gate,
                sinks,
                config.gate_row_stride,
                params,
                suffix_length * num_q_heads,
                command_buffer,
            );
            return Ok(output);
        }
        // Minimize padded query work, then prefer the largest tile among ties.
        // This covers small batches without model-specific GQA special cases.
        let tile = self
            .tiles
            .iter()
            .min_by_key(|tile| (grouped_rows.div_ceil(tile.rows) * tile.rows, std::cmp::Reverse(tile.rows)))
            .unwrap();
        let query_rows_per_threadgroup = tile.rows;
        let query_tiles = grouped_rows.div_ceil(query_rows_per_threadgroup);
        // Below half of the smallest matrix tile, vector work avoids padding.
        let vector = self.vector.as_ref().filter(|_| grouped_rows <= 4);
        let vector_queries = if grouped_rows.is_multiple_of(2) {
            2
        } else {
            1
        };
        let padded_rows = if vector.is_some() {
            grouped_rows
        } else {
            query_tiles * query_rows_per_threadgroup
        };
        let split_rows = num_groups * padded_rows;
        let num_splits = if vector.is_some() {
            // Four independent SIMD groups per threadgroup; target sixteen
            // SIMD groups per core to hide packed-cache load latency.
            (command_buffer.context().gpu_core_count.max(1) * 4)
                .div_ceil(split_rows.div_ceil(4 * vector_queries))
                .clamp(1, params.nk)
        } else if self.use_mxu {
            int8_splits(split_rows, head_dim, suffix_length, kv_length)
        } else {
            int8_simd_splits(
                split_rows,
                head_dim,
                suffix_length,
                kv_length,
                num_groups * query_tiles,
                tile.simdgroups,
                command_buffer.context().gpu_core_count,
            )
        };

        let rows = split_rows * num_splits;
        let mut partials = command_buffer.allocate_scratch_for_shape(&[rows, head_dim], DataType::BF16)?;
        let mut partial_maxs = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        let mut partial_sums = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        if let Some(vector) = vector {
            vector.split[(vector_queries - 1) as usize][usize::from(trie.is_some())].encode(
                input.query_codes,
                input.query_scales,
                input.k_codes,
                input.k_scales,
                input.v_codes,
                input.v_scales,
                &mut partials,
                &mut partial_maxs,
                &mut partial_sums,
                trie,
                cache.ring_params(),
                config.sliding_window_size,
                params,
                split_rows,
                num_splits,
                command_buffer,
            );
        } else {
            let mut dispatch = |kernel: &AttentionGemmGroupedMetalKernel, trie, split_base: u32, split_count: u32| {
                // Disjoint write ranges let the hazard tracker overlap
                // the prefix and suffix dispatches without a barrier.
                let split_byte_range = |row_bytes: usize| {
                    let split_bytes = split_rows as usize * row_bytes;
                    split_base as usize * split_bytes..(split_base + split_count) as usize * split_bytes
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
                    (&mut partials).subrange_mut(split_byte_range(head_dim as usize * DataType::BF16.size_in_bytes())),
                    (&mut partial_maxs).subrange_mut(split_byte_range(DataType::F32.size_in_bytes())),
                    (&mut partial_sums).subrange_mut(split_byte_range(DataType::F32.size_in_bytes())),
                    params,
                    query_tiles,
                    num_splits,
                    split_base,
                    split_count,
                    suffix_length,
                    command_buffer,
                );
            };
            // NAX benefits from specializing fully visible prefix splits.
            // SIMD skips their masks at runtime and needs only one dispatch.
            let prefix_splits = if let Some(prefix) = &tile.prefix {
                let count = (0..num_splits)
                    .take_while(|split| ((split + 1) * params.nk / num_splits) * kv_layout::TILE_KEYS <= params.q_off)
                    .count() as u32;
                if count > 0 {
                    dispatch(prefix, None, 0, count);
                }
                count
            } else {
                0
            };
            dispatch(trie.map_or(&tile.masked, |_| &tile.trie), trie, prefix_splits, num_splits - prefix_splits);
        }

        let mut output =
            command_buffer.allocate_scratch_for_shape(&[suffix_length, num_q_heads, head_dim], config.data_type)?;
        self.combine.encode(
            &partials,
            &partial_maxs,
            &partial_sums,
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

impl AttentionKernel for AttentionMetalKernel {
    type Backend = Metal;
    type Error = AttentionMetalError;

    fn kv_layout(&self) -> Int8KVLayout {
        Int8KVLayout::new(self.config.head_dim, self.config.num_groups, self.use_mxu)
    }

    fn new(
        context: &MetalContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, AttentionMetalError> {
        if !Self::is_supported(&config) {
            return Err(AttentionMetalError::UnsupportedInt8KV);
        }
        let use_mxu = context.supports_mxu && policy::supports_mxu(&config) && config.is_causal;
        Self::new_with_mxu(context, &config, use_mxu).map_err(AttentionMetalError::Backend)
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
    ) -> Result<<Metal as Backend>::ScratchBuffer, AttentionMetalError> {
        let command_buffer = &mut *command_buffer.debug_group("attention core");
        self.encode_attention(input, arguments, command_buffer).map_err(AttentionMetalError::Backend)
    }
}

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/attention/mod.rs"]
mod tests;

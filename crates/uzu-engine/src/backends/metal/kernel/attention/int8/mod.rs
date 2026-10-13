use super::gemm_grouped::policy::{MAX_TRIE_SUFFIX, int8_splits};
use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding,
            gpu_types::{AttnParams, kv_layout},
            kernel::{AttentionKernelConfig, Int8AttentionArguments},
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            kernel::{AttentionGemmGroupedCombineMetalKernel, AttentionGemmGroupedInt8MetalKernel},
        },
    },
    data_type::DataType,
};

/// Grouped rows of one threadgroup: two bands of [`kv_layout::QUERY_TILE_ROWS`].
const BLOCK_ROWS: u32 = 2 * kv_layout::QUERY_TILE_ROWS;

pub(super) struct Int8Attention {
    config: AttentionKernelConfig,
    mask_free: AttentionGemmGroupedInt8MetalKernel,
    masked: AttentionGemmGroupedInt8MetalKernel,
    trie: AttentionGemmGroupedInt8MetalKernel,
    combine: AttentionGemmGroupedCombineMetalKernel,
}

impl Int8Attention {
    pub fn is_supported(
        config: &AttentionKernelConfig,
        context: &MetalContext,
    ) -> bool {
        context.supports_mxu
            && config.head_dim == 256 // TODO: support + tune other head_dim
            && config.is_causal
            && config.data_type == DataType::BF16
            && !config.has_sinks
            && config.sliding_window_size.is_none()
    }

    pub fn new(
        context: &MetalContext,
        config: &AttentionKernelConfig,
    ) -> Result<Self, MetalError> {
        let kernel = |mask_free, is_trie| {
            AttentionGemmGroupedInt8MetalKernel::new(
                context,
                DataType::BF16,
                mask_free,
                config.head_dim,
                is_trie,
                config.num_groups,
            )
        };
        Ok(Self {
            config: *config,
            mask_free: kernel(true, false)?,
            masked: kernel(false, false)?,
            trie: kernel(false, true)?,
            combine: AttentionGemmGroupedCombineMetalKernel::new(
                context,
                DataType::BF16,
                config.head_dim,
                config.gate_row_stride.is_some(),
                true,
            )?,
        })
    }

    pub fn encode(
        &self,
        arguments: Int8AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let Int8AttentionArguments {
            query_codes,
            query_scales,
            kv,
            suffix_length,
            trie,
            cache,
            gate,
        } = arguments;
        let config = &self.config;
        let (head_dim, num_q_heads, num_groups) = (config.head_dim, config.num_q_heads, config.num_groups);
        assert!(trie.is_none() || suffix_length <= MAX_TRIE_SUFFIX, "trie suffix is not supported");
        let prefix_len = cache.prefix_len();
        let kv_length = prefix_len + suffix_length;

        let padded = suffix_length.next_multiple_of(kv_layout::QUERY_TILE_ROWS);
        let gqa_factor = num_q_heads / num_groups;
        let grouped_rows = gqa_factor * padded;
        let m_tiles = grouped_rows.div_ceil(BLOCK_ROWS);
        let split_rows = num_groups * m_tiles * BLOCK_ROWS;
        let num_splits = int8_splits(split_rows, head_dim, suffix_length, kv_length);

        let rows = split_rows * num_splits;
        let mut partials = command_buffer.allocate_scratch_for_shape(&[rows, head_dim], DataType::BF16)?;
        let mut partial_maxs = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        let mut partial_sums = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        let params = AttnParams {
            q_strides: [num_q_heads * padded * head_dim, padded * head_dim, head_dim],
            k_strides: [0, head_dim, num_groups * head_dim],
            v_strides: [0, head_dim, num_groups * head_dim],
            o_strides: [suffix_length * num_q_heads * head_dim, head_dim, num_q_heads * head_dim],
            gqa_factor,
            scale: config.scale.unwrap_or(1.0 / (head_dim as f32).sqrt()),
            q_len: padded,
            k_len: kv_length,
            q_off: prefix_len,
            nk: kv_length.div_ceil(kv_layout::TILE_KEYS),
            nk_aligned: kv_length / kv_layout::TILE_KEYS,
            k_rem: kv_length % kv_layout::TILE_KEYS,
            ..AttnParams::default()
        };
        let mask_free_splits = (0..num_splits)
            .take_while(|split| ((split + 1) * params.nk / num_splits) * kv_layout::TILE_KEYS <= params.q_off)
            .count() as u32;
        let mut dispatch = |kernel: &AttentionGemmGroupedInt8MetalKernel, trie, split_base, split_count| {
            kernel.encode(
                query_codes,
                query_scales,
                &*kv.k_codes,
                &kv.k_scales,
                &*kv.v_codes,
                &kv.v_scales,
                trie,
                &mut partials,
                &mut partial_maxs,
                &mut partial_sums,
                params,
                m_tiles,
                1,
                num_splits,
                kv.layout.padded_capacity,
                split_base,
                split_count,
                suffix_length,
                command_buffer,
            )
        };
        if mask_free_splits > 0 {
            dispatch(&self.mask_free, None, 0, mask_free_splits);
        }
        dispatch(trie.map_or(&self.masked, |_| &self.trie), trie, mask_free_splits, num_splits - mask_free_splits);

        let mut output =
            command_buffer.allocate_scratch_for_shape(&[suffix_length, num_q_heads, head_dim], DataType::BF16)?;
        self.combine.encode(
            &partials,
            &partial_maxs,
            &partial_sums,
            &mut output,
            gate,
            params,
            grouped_rows,
            BLOCK_ROWS,
            m_tiles,
            num_groups,
            1,
            num_splits,
            config.gate_row_stride,
            Some(suffix_length),
            command_buffer,
        );
        Ok(output)
    }
}

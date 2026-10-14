use super::gemm_grouped::{
    block_rows,
    policy::{self, MAX_TRIE_SUFFIX, int8_splits},
};
use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding,
            gpu_types::kv_layout,
            kernel::{AttentionArguments, AttentionKernelConfig, Int8Input},
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            gpu_types::attention::AttnParams,
            kernel::{AttentionGemmGroupedInt8CombineMetalKernel, AttentionGemmGroupedInt8MetalKernel},
        },
    },
    data_type::DataType,
};

pub struct Int8Attention {
    config: AttentionKernelConfig,
    mask_free: AttentionGemmGroupedInt8MetalKernel,
    masked: AttentionGemmGroupedInt8MetalKernel,
    trie: AttentionGemmGroupedInt8MetalKernel,
    combine: AttentionGemmGroupedInt8CombineMetalKernel,
}

impl Int8Attention {
    pub fn is_supported(
        config: &AttentionKernelConfig,
        context: &MetalContext,
    ) -> bool {
        policy::is_supported(config, context) && config.is_causal
    }

    pub fn new(
        context: &MetalContext,
        config: &AttentionKernelConfig,
    ) -> Result<Self, MetalError> {
        let kernel = |mask_free, is_trie| {
            AttentionGemmGroupedInt8MetalKernel::new(
                context,
                DataType::BF16,
                config.head_dim,
                mask_free,
                is_trie,
                config.num_groups,
            )
        };
        Ok(Self {
            config: *config,
            mask_free: kernel(true, false)?,
            masked: kernel(false, false)?,
            trie: kernel(false, true)?,
            combine: AttentionGemmGroupedInt8CombineMetalKernel::new(
                context,
                DataType::BF16,
                config.head_dim,
                config.gate_row_stride.is_some(),
            )?,
        })
    }

    pub fn encode(
        &self,
        input: Int8Input<
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        arguments: AttentionArguments<'_, Metal, impl BufferRef<Backend = Metal>, impl BufferRef<Backend = Metal>>,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        let AttentionArguments {
            suffix_length,
            trie,
            cache,
            gate,
            ..
        } = arguments;
        let config = &self.config;
        let (head_dim, num_q_heads, num_groups) = (config.head_dim, config.num_q_heads, config.num_groups);
        assert!(trie.is_none() || suffix_length <= MAX_TRIE_SUFFIX, "trie suffix is not supported");
        let prefix_len = cache.prefix_len();
        let kv_length = prefix_len + suffix_length;

        let gqa_factor = num_q_heads / num_groups;
        let grouped_rows = gqa_factor * suffix_length;
        let block_rows = block_rows(head_dim);
        let query_tiles = grouped_rows.div_ceil(block_rows);
        let split_rows = num_groups * query_tiles * block_rows;
        let num_splits = int8_splits(split_rows, head_dim, suffix_length, kv_length);

        let rows = split_rows * num_splits;
        let mut partials = command_buffer.allocate_scratch_for_shape(&[rows, head_dim], DataType::BF16)?;
        let mut partial_maxs = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
        let mut partial_sums = command_buffer.allocate_scratch_for_shape(&[rows], DataType::F32)?;
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
            nk_aligned: kv_length / kv_layout::TILE_KEYS,
            k_rem: kv_length % kv_layout::TILE_KEYS,
            ..AttnParams::default()
        };
        let prefix_splits = (0..num_splits)
            .take_while(|split| ((split + 1) * params.nk / num_splits) * kv_layout::TILE_KEYS <= params.q_off)
            .count() as u32;
        let mut dispatch = |kernel: &AttentionGemmGroupedInt8MetalKernel, trie, split_base: u32, split_count: u32| {
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
            )
        };
        if prefix_splits > 0 {
            dispatch(&self.mask_free, None, 0, prefix_splits);
        }
        dispatch(trie.map_or(&self.masked, |_| &self.trie), trie, prefix_splits, num_splits - prefix_splits);

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
            query_tiles * block_rows,
            num_groups,
            num_splits,
            config.gate_row_stride,
            command_buffer,
        );
        Ok(output)
    }
}

use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding,
            kernel::{AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, kv_layout::KVLayout},
        },
        webgpu::{
            WebGPU, buffer::binding_parts, command_buffer::WebGPUCommandBufferEncoding, context::WebGPUContext,
            error::WebGPUError,
        },
    },
    data_type::DataType,
};

include!(concat!(env!("OUT_DIR"), "/webgpu/attention/compute.rs"));

pub struct WebGPUAttention {
    config: AttentionKernelConfig,
}

impl AttentionKernel for WebGPUAttention {
    type Backend = WebGPU;
    type Error = WebGPUError;

    fn new(
        _context: &WebGPUContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, Self::Error> {
        if !matches!(config.head_dim, 64 | 128 | 256 | 512)
            || !matches!(config.data_type, DataType::F32 | DataType::BF16)
            || config.num_groups == 0
            || config.num_q_heads == 0
            || !config.num_q_heads.is_multiple_of(config.num_groups)
        {
            return Err(WebGPUError::UnsupportedKernelVariant("attention shape or data type".into()));
        }
        Ok(Self {
            config,
        })
    }

    fn kv_layout(&self) -> KVLayout {
        KVLayout::new(self.config.head_dim, self.config.num_groups, 1, 0)
    }

    fn encode(
        &self,
        input: AttentionInput<
            impl BufferRef<Backend = WebGPU>,
            impl BufferRef<Backend = WebGPU>,
            impl BufferRef<Backend = WebGPU>,
        >,
        arguments: AttentionArguments<'_, WebGPU, impl BufferRef<Backend = WebGPU>, impl BufferRef<Backend = WebGPU>>,
        command: &mut WebGPUCommandBufferEncoding,
    ) -> Result<<WebGPU as Backend>::ScratchBuffer, Self::Error> {
        let config = self.config;
        let gate_stride = config.gate_row_stride.filter(|_| arguments.gate.is_some());
        let mut output = command.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, config.head_dim],
            config.data_type,
        )?;
        let kernel = AttentionSimpleWebGPUKernel::new(
            command.context(),
            config.data_type,
            config.head_dim,
            config.is_causal,
            arguments.trie.is_some(),
            arguments.cache.ring_params().is_some(),
            config.sliding_window_size.is_some(),
            arguments.sinks.is_some(),
            gate_stride.is_some(),
        )?;
        kernel.encode(
            input.query_codes,
            input.query_scales,
            input.k_codes,
            input.k_scales,
            input.v_codes,
            input.v_scales,
            arguments.trie,
            arguments.cache.ring_params(),
            config.sliding_window_size,
            arguments.sinks,
            arguments.gate.filter(|_| gate_stride.is_some()),
            gate_stride,
            &mut output,
            arguments.cache.prefix_len(),
            arguments.suffix_length,
            config.num_q_heads,
            config.num_groups,
            config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
            command,
        );
        Ok(output)
    }
}

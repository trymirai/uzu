use crate::backends::{
    common::{
        Backend, BufferRef, CommandBufferEncoding,
        kernel::{
            AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, AttentionSinglePassKernel,
            SigmoidGateKernel,
        },
    },
    webgpu::{
        WebGPU,
        command_buffer::WebGPUCommandBufferEncoding,
        context::WebGPUContext,
        error::WebGPUError,
        kernel::{AttentionSinglePassWebGPUKernel, SigmoidGateWebGPUKernel},
    },
};

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
        if !matches!(config.head_dim, 64 | 128 | 256 | 512) {
            return Err(WebGPUError::UnsupportedKernelVariant(format!("attention head dimension {}", config.head_dim)));
        }
        Ok(Self {
            config,
        })
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
        let AttentionInput::FullPrecision(input) = input else {
            return Err(WebGPUError::NotSupported);
        };
        if arguments.trie.is_some() {
            return Err(WebGPUError::NotSupported);
        }
        let c = self.config;
        let mut output =
            command.allocate_scratch_for_shape(&[arguments.suffix_length, c.num_q_heads, c.head_dim], c.data_type)?;
        let kernel = AttentionSinglePassWebGPUKernel::new(
            command.context(),
            c.data_type,
            c.head_dim,
            c.has_sinks,
            c.is_kv_cache_ring,
            c.is_causal,
            false,
            c.sliding_window_size.is_some(),
        )?;
        kernel.encode(
            input.queries,
            input.keys,
            input.values,
            &mut output,
            c.num_q_heads / c.num_groups,
            arguments.cache.prefix_len() + arguments.suffix_length,
            c.head_dim,
            c.num_groups * c.head_dim,
            c.head_dim,
            c.num_groups * c.head_dim,
            arguments.cache.ring_params(),
            c.scale.unwrap_or(1.0 / (c.head_dim as f32).sqrt()),
            arguments.trie,
            c.sliding_window_size,
            arguments.sinks,
            c.num_q_heads,
            arguments.suffix_length,
            command,
        );
        if let (Some(gate), Some(stride)) = (arguments.gate, c.gate_row_stride) {
            SigmoidGateWebGPUKernel::new(command.context(), c.data_type)?.encode(
                gate,
                &mut output,
                c.num_q_heads * c.head_dim,
                arguments.suffix_length,
                stride,
                command,
            );
        }
        Ok(output)
    }
}

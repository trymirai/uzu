use crate::backends::common::{
    Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
    kernel::{AttentionArguments, AttentionKernelConfig, AttentionSinglePassKernel, SigmoidGateKernel},
};

pub struct SinglePassAttention<B: Backend> {
    config: AttentionKernelConfig,
    single_pass: <B::Kernels as Kernels>::AttentionSinglePassKernel,
    trie_single_pass: <B::Kernels as Kernels>::AttentionSinglePassKernel,
    gate: Option<<B::Kernels as Kernels>::SigmoidGateKernel>,
}

impl<B: Backend> SinglePassAttention<B> {
    pub fn create(
        context: &B::Context,
        config: AttentionKernelConfig,
    ) -> Result<Self, B::Error> {
        let single_pass = |is_trie| {
            <<B::Kernels as Kernels>::AttentionSinglePassKernel as AttentionSinglePassKernel>::new(
                context,
                config.data_type,
                config.head_dim,
                config.has_sinks,
                config.is_kv_cache_ring,
                config.is_causal,
                is_trie,
                config.sliding_window_size.is_some(),
            )
        };
        Ok(Self {
            config,
            single_pass: single_pass(false)?,
            trie_single_pass: single_pass(true)?,
            gate: config
                .gate_row_stride
                .map(|_| {
                    <<B::Kernels as Kernels>::SigmoidGateKernel as SigmoidGateKernel>::new(context, config.data_type)
                })
                .transpose()?,
        })
    }

    /// AttentionSinglePass, then the sigmoid gate where the call has one.
    pub fn encode_gated(
        &self,
        arguments: AttentionArguments<
            '_,
            B,
            impl BufferRef<Backend = B>,
            impl BufferRef<Backend = B>,
            impl BufferRef<Backend = B>,
            impl BufferRef<Backend = B>,
            impl BufferRef<Backend = B>,
        >,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let config = self.config;
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, config.head_dim],
            config.data_type,
        )?;
        let single_pass = if arguments.trie.is_some() {
            &self.trie_single_pass
        } else {
            &self.single_pass
        };
        {
            let command_buffer = &mut *command_buffer.debug_group("attention core");
            single_pass.encode(
                arguments.queries,
                arguments.keys,
                arguments.values,
                &mut output,
                config.num_q_heads / config.num_groups,
                arguments.cache.prefix_len() + arguments.suffix_length,
                config.head_dim,
                config.num_groups * config.head_dim,
                config.head_dim,
                config.num_groups * config.head_dim,
                arguments.cache.ring_params(),
                config.scale.unwrap_or(1.0 / (config.head_dim as f32).sqrt()),
                arguments.trie,
                config.sliding_window_size,
                arguments.sinks,
                config.num_q_heads,
                arguments.suffix_length,
                command_buffer,
            );
        }
        if let (Some(gate), Some(gate_kernel), Some(gate_row_stride)) =
            (arguments.gate, &self.gate, config.gate_row_stride)
        {
            gate_kernel.encode(
                gate,
                &mut output,
                config.num_q_heads * config.head_dim,
                arguments.suffix_length,
                gate_row_stride,
                command_buffer,
            );
        }
        Ok(output)
    }
}

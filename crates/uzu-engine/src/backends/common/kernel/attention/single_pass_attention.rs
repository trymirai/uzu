use crate::backends::common::{
    Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
    kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig, AttentionSinglePassKernel},
};

pub struct SinglePassAttention<B: Backend> {
    config: AttentionKernelConfig,
    single_pass: <B::Kernels as Kernels>::AttentionSinglePassKernel,
    trie_single_pass: <B::Kernels as Kernels>::AttentionSinglePassKernel,
}

impl<B: Backend<Kernels: Kernels<AttentionKernel = SinglePassAttention<B>>>> AttentionKernel
    for SinglePassAttention<B>
{
    type Backend = B;

    fn new(
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
        })
    }

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            B,
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
        command_buffer.push_debug_group("attention core");
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
        command_buffer.pop_debug_group();
        Ok(output)
    }
}

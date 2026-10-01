use crate::backends::common::Backend;

pub mod ancestor_attention;
pub mod attention_prepare;
pub mod attention_single_pass;
pub mod kv_cache_update;
pub mod qkv_norm;
pub mod sigmoid_gate;

mod mask;

use crate::backends::{
    common::{
        BufferRef, CommandBufferEncoding,
        kernel::{AttentionArguments, AttentionKernel, AttentionKernelConfig, AttentionSinglePassKernel},
    },
    cpu::{Cpu, command_buffer::CpuCommandBufferEncoding, context::CpuContext, error::CpuError},
};

pub struct AttentionCpuKernel {
    config: AttentionKernelConfig,
}

impl AttentionKernel for AttentionCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &CpuContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, CpuError> {
        Ok(Self {
            config,
        })
    }

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Cpu,
            impl BufferRef<Backend = Cpu>,
            impl BufferRef<Backend = Cpu>,
            impl BufferRef<Backend = Cpu>,
            impl BufferRef<Backend = Cpu>,
        >,
        command_buffer: &mut CpuCommandBufferEncoding,
    ) -> Result<<Cpu as Backend>::ScratchBuffer, CpuError> {
        let config = self.config;
        let mut output = command_buffer.allocate_scratch_for_shape(
            &[arguments.suffix_length, config.num_q_heads, config.head_dim],
            config.data_type,
        )?;

        let single_pass = <attention_single_pass::AttentionSinglePassCpuKernel as AttentionSinglePassKernel>::new(
            command_buffer.context(),
            config.data_type,
            config.head_dim,
            config.has_sinks,
            config.is_kv_cache_ring,
            config.is_causal,
            arguments.trie.is_some(),
            config.sliding_window_size.is_some(),
        )?;
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

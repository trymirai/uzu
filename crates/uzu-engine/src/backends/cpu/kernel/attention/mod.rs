use crate::backends::common::Backend;

pub mod ancestor_attention;
pub mod kv_cache_update;
pub mod prepare;

mod compute;
mod mask;
mod rope;

use crate::backends::{
    common::{
        BufferRef,
        kernel::{AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, kv_layout::KVLayout},
    },
    cpu::{Cpu, command_buffer::CpuCommandBufferEncoding, context::CpuContext, error::CpuError},
};

pub struct AttentionCpuKernel {
    config: AttentionKernelConfig,
}

impl AttentionKernel for AttentionCpuKernel {
    type Backend = Cpu;
    type Error = CpuError;

    fn new(
        _context: &CpuContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, CpuError> {
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
            impl BufferRef<Backend = Cpu>,
            impl BufferRef<Backend = Cpu>,
            impl BufferRef<Backend = Cpu>,
        >,
        arguments: AttentionArguments<'_, Cpu, impl BufferRef<Backend = Cpu>, impl BufferRef<Backend = Cpu>>,
        command_buffer: &mut CpuCommandBufferEncoding,
    ) -> Result<<Cpu as Backend>::ScratchBuffer, CpuError> {
        compute::encode(input, arguments, self.config, command_buffer)
    }
}

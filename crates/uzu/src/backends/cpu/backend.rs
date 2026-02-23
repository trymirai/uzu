use crate::backends::{
    common::Backend,
    cpu::{
        buffer::CpuBuffer, command_buffer::CpuCommandBuffer, context::CpuContext, error::CpuError, event::CpuEvent,
        kernel::CpuKernels,
    },
};

#[derive(Debug, Clone)]
pub struct Cpu;

impl Backend for Cpu {
    type Context = CpuContext;
    type NativeBuffer = CpuBuffer;
    type CommandBuffer = CpuCommandBuffer;
    type ComputeEncoder = CpuCommandBuffer;
    type CopyEncoder = CpuCommandBuffer;
    type Event = CpuEvent;
    type Kernels = CpuKernels;
    type Error = CpuError;

    const MAX_INLINE_BYTES: usize = usize::MAX;
}

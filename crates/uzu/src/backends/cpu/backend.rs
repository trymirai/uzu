use crate::backends::{
    common::Backend,
    cpu::{
        buffer::CpuBuffer, command_buffer::CpuCommandBuffer, compute_encoder::CpuComputeEncoder, context::CpuContext,
        copy_encoder::CpuCopyEncoder, error::CpuError, event::CpuEvent, kernel::CpuKernels,
    },
};

#[derive(Debug, Clone)]
pub struct Cpu;

impl Backend for Cpu {
    type Context = CpuContext;
    type NativeBuffer = CpuBuffer;
    type CommandBuffer = CpuCommandBuffer;
    type ComputeEncoder = CpuComputeEncoder;
    type CopyEncoder = CpuCopyEncoder;
    type Event = CpuEvent;
    type Kernels = CpuKernels;
    type Error = CpuError;
}

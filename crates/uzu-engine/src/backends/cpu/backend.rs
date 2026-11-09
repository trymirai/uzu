use std::convert::Infallible;

use crate::backends::{
    common::{
        Backend,
        allocator::{
            bump::BumpAllocation,
            pool::{PoolAllocation, PoolAllocator},
        },
    },
    cpu::{
        buffer::{dense::CpuDenseBuffer, sparse::CpuSparseBuffer},
        command_buffer::{CpuCommandBuffer, CpuCommandBufferEncoding},
        context::CpuContext,
        error::CpuError,
        kernel::CpuKernels,
    },
};

#[derive(Debug, Clone)]
pub struct Cpu;

impl Backend for Cpu {
    type Context = CpuContext;
    type CommandBuffer = CpuCommandBuffer;
    type DebugGroupSpan<'a> = &'a mut CpuCommandBufferEncoding;
    type GlobalGpuBuffer = CpuDenseBuffer;
    type GlobalBufferCpuGpu = CpuDenseBuffer;
    type ConstantBuffer = BumpAllocation<CpuDenseBuffer>;
    type ScratchBuffer = PoolAllocation<CpuDenseBuffer>;
    type SparseBuffer = CpuSparseBuffer;
    type AllocationPool = PoolAllocator<CpuDenseBuffer>;
    type CaptureSpan = Infallible;
    type Kernels = CpuKernels;
    type Error = CpuError;

    const NAME: &'static str = "cpu";
}

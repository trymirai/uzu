use crate::backends::{
    common::{
        Backend,
        allocator::{
            block::BlockAllocation,
            bump::BumpAllocation,
            pool::{PoolAllocation, PoolAllocator},
        },
    },
    metal::{
        buffer::{dense::MetalDenseBuffer, sparse::MetalSparseBuffer},
        capture::MetalCaptureSpan,
        command_buffer::{MetalCommandBuffer, MetalDebugGroupSpan},
        context::MetalContext,
        error::MetalError,
        kernel::MetalKernels,
    },
};

#[derive(Debug, Clone)]
pub struct Metal;

impl Backend for Metal {
    type Context = MetalContext;
    type CommandBuffer = MetalCommandBuffer;
    type DebugGroupSpan<'a> = MetalDebugGroupSpan<'a>;
    type GlobalGpuBuffer = BlockAllocation<MetalDenseBuffer>;
    type GlobalBufferCpuGpu = BlockAllocation<MetalDenseBuffer>;
    type ConstantBuffer = BumpAllocation<Self::GlobalGpuBuffer>;
    type ScratchBuffer = PoolAllocation<Self::GlobalGpuBuffer>;
    type SparseBuffer = MetalSparseBuffer;
    type AllocationPool = PoolAllocator<Self::GlobalGpuBuffer>;
    type CaptureSpan = MetalCaptureSpan;
    type Kernels = MetalKernels;
    type Error = MetalError;

    const NAME: &'static str = "metal";
}

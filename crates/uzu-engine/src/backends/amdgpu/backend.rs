use crate::backends::{
    amdgpu::{
        buffer::{constant::ConstantPage, dense::AmdgpuBuffer, device::ScratchPage, sparse::AmdgpuSparseBuffer},
        command_buffer::AmdgpuCommandBuffer,
        context::AmdgpuContext,
        error::AmdgpuError,
        kernel::AmdgpuKernels,
    },
    common::{
        Backend,
        allocator::{
            bump::BumpAllocation,
            pool::{PoolAllocation, PoolAllocator},
        },
    },
};

#[derive(Debug, Clone)]
pub struct Amdgpu;

impl Backend for Amdgpu {
    type Context = AmdgpuContext;
    type CommandBuffer = AmdgpuCommandBuffer;
    type GlobalBuffer = AmdgpuBuffer;
    type ConstantBuffer = BumpAllocation<ConstantPage>;
    type ScratchBuffer = PoolAllocation<ScratchPage>;
    type SparseBuffer = AmdgpuSparseBuffer;
    type AllocationPool = PoolAllocator<ScratchPage>;
    type Kernels = AmdgpuKernels;
    type Error = AmdgpuError;

    const NAME: &'static str = "amdgpu";
}

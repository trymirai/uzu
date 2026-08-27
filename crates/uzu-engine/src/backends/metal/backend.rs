use crate::backends::{
    common::{
        Backend,
        allocator::{Allocation, AllocationPool},
    },
    metal::{
        buffer::{dense::MetalDenseBuffer, sparse::MetalSparseBuffer},
        command_buffer::MetalCommandBuffer,
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
    type GlobalBuffer = Allocation<MetalDenseBuffer>;
    type ConstantBuffer = Allocation<MetalDenseBuffer>;
    type ScratchBuffer = Allocation<MetalDenseBuffer>;
    type SparseBuffer = MetalSparseBuffer;
    type AllocationPool = AllocationPool<MetalDenseBuffer>;
    type Kernels = MetalKernels;
    type Error = MetalError;

    const NAME: &'static str = "metal";
}

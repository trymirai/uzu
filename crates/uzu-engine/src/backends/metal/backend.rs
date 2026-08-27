use crate::backends::{
    common::Backend,
    metal::{
        command_buffer::MetalCommandBuffer, context::MetalContext, dense_buffer::MetalDenseBuffer, error::MetalError,
        kernel::MetalKernels, sparse::MetalSparseBuffer,
    },
};

#[derive(Debug, Clone)]
pub struct Metal;

impl Backend for Metal {
    type Context = MetalContext;
    type CommandBuffer = MetalCommandBuffer;
    type DenseBuffer = MetalDenseBuffer;
    type SparseBuffer = MetalSparseBuffer;
    type Kernels = MetalKernels;
    type Error = MetalError;

    const NAME: &'static str = "metal";
    const MIN_ALLOCATION_ALIGNMENT: usize = 4;
    const MAX_ALLOCATION_ALIGNMENT: usize = 64;
    const ALLOCATION_GRANULARITY: usize = 8 * 1024 * 1024;
}

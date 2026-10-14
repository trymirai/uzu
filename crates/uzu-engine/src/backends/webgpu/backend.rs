use std::convert::Infallible;

use super::{
    buffer::{WebGpuBuffer, WebGpuSparseBuffer},
    command_buffer::{WebGpuCommandBuffer, WebGpuCommandBufferEncoding},
    context::WebGpuContext,
    error::WebGpuError,
    kernel::WebGpuKernels,
};
use crate::backends::common::{
    Backend,
    allocator::{
        bump::BumpAllocation,
        pool::{PoolAllocation, PoolAllocator},
    },
};

#[derive(Debug, Clone)]
pub struct WebGpu;

impl Backend for WebGpu {
    type Context = WebGpuContext;
    type CommandBuffer = WebGpuCommandBuffer;
    type DebugGroupSpan<'a> = &'a mut WebGpuCommandBufferEncoding;
    type GlobalBuffer = WebGpuBuffer;
    type ConstantBuffer = BumpAllocation<WebGpuBuffer>;
    type ScratchBuffer = PoolAllocation<WebGpuBuffer>;
    type SparseBuffer = WebGpuSparseBuffer;
    type AllocationPool = PoolAllocator<WebGpuBuffer>;
    type CaptureSpan = Infallible;
    type Kernels = WebGpuKernels;
    type Error = WebGpuError;

    const NAME: &'static str = "webgpu";
}

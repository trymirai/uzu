use std::convert::Infallible;

use crate::backends::{
    common::{
        Backend,
        allocator::{
            bump::BumpAllocation,
            pool::{PoolAllocation, PoolAllocator},
        },
    },
    webgpu::{
        buffer::{
            dense::{WebGPUDenseBufferCpuGpu, WebGPUDenseBufferGpu},
            sparse::WebGPUSparseBuffer,
        },
        command_buffer::{WebGPUCommandBuffer, WebGPUCommandBufferEncoding},
        context::WebGPUContext,
        error::WebGPUError,
        kernel::WebGPUKernels,
    },
};

#[derive(Debug, Clone)]
pub struct WebGPU;

impl Backend for WebGPU {
    type Context = WebGPUContext;
    type CommandBuffer = WebGPUCommandBuffer;
    type DebugGroupSpan<'a> = &'a mut WebGPUCommandBufferEncoding;
    type GlobalGpuBuffer = WebGPUDenseBufferGpu;
    type GlobalBufferCpuGpu = WebGPUDenseBufferCpuGpu;
    type ConstantBuffer = BumpAllocation<WebGPUDenseBufferCpuGpu>;
    type ScratchBuffer = PoolAllocation<WebGPUDenseBufferGpu>;
    type AllocationPool = PoolAllocator<WebGPUDenseBufferGpu>;
    type CaptureSpan = Infallible;
    type SparseBuffer = WebGPUSparseBuffer;
    type Kernels = WebGPUKernels;
    type Error = WebGPUError;

    const NAME: &'static str = "webgpu";
}

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
            constant::WebGPUConstantBuffer,
            global::{WebGPUGlobalBuffer, WebGPUGlobalBufferInitializer},
            scratch::WebGPUScratchBuffer,
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
    type GlobalBuffer = WebGPUGlobalBuffer;
    type GlobalBufferInitializer = WebGPUGlobalBufferInitializer;
    type ConstantBuffer = BumpAllocation<WebGPUConstantBuffer>;
    type ScratchBuffer = PoolAllocation<WebGPUScratchBuffer>;
    type AllocationPool = PoolAllocator<WebGPUScratchBuffer>;
    type CaptureSpan = Infallible;
    type SparseBuffer = WebGPUSparseBuffer;
    type Kernels = WebGPUKernels;
    type Error = WebGPUError;

    const NAME: &'static str = "webgpu";
}

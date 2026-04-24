use crate::backends::{
    common::Backend,
    webgpu::{
        buffer::WebGPUBuffer, command_buffer::WebGPUCommandBuffer, context::WebGPUContext, error::WebGPUError,
        event::WebGPUEvent, kernel::WebGPUKernels,
    },
};

#[derive(Debug, Clone)]
pub struct WebGPU;

impl Backend for WebGPU {
    type Context = WebGPUContext;
    type CommandBuffer = WebGPUCommandBuffer;
    type Buffer = WebGPUBuffer;
    type Event = WebGPUEvent;
    type Kernels = WebGPUKernels;
    type Error = WebGPUError;

    // Copy-pasted from metal, TODO: should be runtime
    const MIN_ALLOCATION_ALIGNMENT: usize = 64;
    const MAX_INLINE_BYTES: usize = 4096;
}

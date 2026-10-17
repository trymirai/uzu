use crate::backends::{
    common::{Buffer, ScratchBuffer, allocator::pool::PoolAllocation},
    webgpu::{WebGPU, context::WebGPUContext, error::WebGPUError},
};

#[derive(Debug)]
pub struct WebGPUScratchBuffer {
    buffer: wgpu::Buffer,
}

impl WebGPUScratchBuffer {
    pub(in crate::backends::webgpu) fn new(
        context: &WebGPUContext,
        size: usize,
    ) -> Result<Self, WebGPUError> {
        let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            buffer,
        })
    }

    pub(super) fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Buffer for PoolAllocation<WebGPUScratchBuffer> {
    type Backend = WebGPU;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl ScratchBuffer for PoolAllocation<WebGPUScratchBuffer> {}

use std::{os::raw::c_void, ptr::NonNull};

use crate::backends::{
    common::{Buffer, BufferCpuAccessible, GlobalBuffer},
    webgpu::{WebGPU, context::WebGPUContext, error::WebGPUError},
};

#[derive(Debug)]
pub struct WebGPUGlobalBuffer {
    buffer: wgpu::Buffer,
}

impl WebGPUGlobalBuffer {
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

impl Buffer for WebGPUGlobalBuffer {
    type Backend = WebGPU;

    fn size(&self) -> usize {
        self.wgpu_buffer().size() as usize
    }
}

impl BufferCpuAccessible for WebGPUGlobalBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        todo!("WebGPU buffer CPU access")
    }
}

impl GlobalBuffer for WebGPUGlobalBuffer {}

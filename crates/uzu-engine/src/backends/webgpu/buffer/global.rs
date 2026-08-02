use std::{os::raw::c_void, ptr::NonNull};

use crate::{
    backends::{
        common::{Buffer, BufferCpuAccessible, GlobalBuffer, GlobalBufferInitializer},
        webgpu::{WebGPU, context::WebGPUContext, error::WebGPUError},
    },
    utils::pointers::SendPtrMut,
};

#[derive(Debug)]
pub struct WebGPUGlobalBufferInitializer {
    cpu_ptr: SendPtrMut<c_void>,
    view: wgpu::BufferViewMut,
    buffer: wgpu::Buffer,
}

impl WebGPUGlobalBufferInitializer {
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
            mapped_at_creation: true,
        });

        let mut view = buffer.get_mapped_range_mut(..)?;
        let cpu_ptr = SendPtrMut(view.slice(..).as_raw_element_ptr().cast().as_ptr());

        Ok(Self {
            buffer,
            view,
            cpu_ptr,
        })
    }
}

impl Buffer for WebGPUGlobalBufferInitializer {
    type Backend = WebGPU;

    fn size(&self) -> usize {
        self.buffer.size() as usize
    }
}

impl BufferCpuAccessible for WebGPUGlobalBufferInitializer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked(self.cpu_ptr.as_ptr()) }
    }
}

impl GlobalBufferInitializer for WebGPUGlobalBufferInitializer {
    fn end_initialization(self) -> <Self::Backend as crate::backends::Backend>::GlobalBuffer {
        drop(self.view);
        self.buffer.unmap();
        WebGPUGlobalBuffer {
            buffer: self.buffer,
        }
    }
}

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

    pub(in crate::backends::webgpu) fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Buffer for WebGPUGlobalBuffer {
    type Backend = WebGPU;

    fn size(&self) -> usize {
        self.buffer.size() as usize
    }
}

impl BufferCpuAccessible for WebGPUGlobalBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unimplemented!()
    }
}

impl GlobalBuffer for WebGPUGlobalBuffer {}

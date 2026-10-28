use std::{mem::ManuallyDrop, os::raw::c_void, ptr::NonNull};

use crate::{
    backends::{
        common::{Buffer, CpuBuffer, GlobalBufferCpuGpu, GlobalGpuBuffer, GpuBuffer, allocator::bump::BumpAllocation},
        webgpu::{WebGPU, context::WebGPUContext, error::WebGPUError},
    },
    utils::pointers::SendPtrMut,
};

#[derive(Debug)]
pub struct WebGPUDenseBufferGpu {
    buffer: wgpu::Buffer,
    size: usize,
}

impl WebGPUDenseBufferGpu {
    pub(in crate::backends::webgpu) fn new(
        context: &WebGPUContext,
        size: usize,
    ) -> Result<Self, WebGPUError> {
        let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize) as u64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            buffer,
            size,
        })
    }

    pub(in crate::backends::webgpu) fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Buffer for WebGPUDenseBufferGpu {
    type Backend = WebGPU;

    fn size(&self) -> usize {
        self.size
    }
}

impl GpuBuffer for WebGPUDenseBufferGpu {}

impl GlobalGpuBuffer for WebGPUDenseBufferGpu {}

#[derive(Debug)]
pub struct WebGPUDenseBufferCpuGpu {
    cpu_ptr: SendPtrMut<c_void>,
    view: ManuallyDrop<wgpu::BufferViewMut>,
    buffer: wgpu::Buffer,
    size: usize,
}

impl WebGPUDenseBufferCpuGpu {
    pub(in crate::backends::webgpu) fn new(
        context: &WebGPUContext,
        size: usize,
    ) -> Result<Self, WebGPUError> {
        let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize) as u64,
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
            cpu_ptr,
            view: ManuallyDrop::new(view),
            buffer,
            size,
        })
    }

    pub(in crate::backends::webgpu) fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Drop for WebGPUDenseBufferCpuGpu {
    fn drop(&mut self) {
        unsafe { ManuallyDrop::drop(&mut self.view) };
        self.buffer.unmap();
    }
}

impl Buffer for WebGPUDenseBufferCpuGpu {
    type Backend = WebGPU;

    fn size(&self) -> usize {
        self.size
    }
}

impl GpuBuffer for BumpAllocation<WebGPUDenseBufferCpuGpu> {}

impl CpuBuffer for WebGPUDenseBufferCpuGpu {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked(self.cpu_ptr.as_ptr()) }
    }
}

impl GlobalBufferCpuGpu for WebGPUDenseBufferCpuGpu {
    fn handoff(self) -> WebGPUDenseBufferGpu {
        let gpu = WebGPUDenseBufferGpu {
            buffer: self.buffer.clone(),
            size: self.size,
        };
        drop(self);
        gpu
    }
}

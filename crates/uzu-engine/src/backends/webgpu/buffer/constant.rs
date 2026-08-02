use std::{mem::ManuallyDrop, os::raw::c_void, ptr::NonNull};

use crate::{
    backends::{
        common::{Buffer, BufferCpuAccessible, ConstantBuffer, allocator::bump::BumpAllocation},
        webgpu::{WebGPU, context::WebGPUContext, error::WebGPUError},
    },
    utils::pointers::SendPtrMut,
};

#[derive(Debug)]
pub struct WebGPUConstantBuffer {
    buffer: wgpu::Buffer,
    view: ManuallyDrop<wgpu::BufferViewMut>,
    cpu_ptr: SendPtrMut<c_void>,
}

impl WebGPUConstantBuffer {
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
            view: ManuallyDrop::new(view),
            cpu_ptr,
        })
    }

    pub(super) fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Drop for WebGPUConstantBuffer {
    fn drop(&mut self) {
        unsafe { ManuallyDrop::drop(&mut self.view) };
        self.buffer.unmap();
    }
}

impl Buffer for BumpAllocation<WebGPUConstantBuffer> {
    type Backend = WebGPU;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl BufferCpuAccessible for BumpAllocation<WebGPUConstantBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked(self.page().cpu_ptr.as_ptr().byte_add(self.range().start)) }
    }
}

impl ConstantBuffer for BumpAllocation<WebGPUConstantBuffer> {}

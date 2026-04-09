use std::{os::raw::c_void, ptr::NonNull};

use crate::backends::{common::Buffer, webgpu::WebGPU};

#[derive(Debug)]
pub struct WebGPUBuffer {
    pub buffer: wgpu::Buffer,
}

impl Buffer for WebGPUBuffer {
    type Backend = WebGPU;

    fn set_label(
        &mut self,
        _label: Option<&str>,
    ) {
        todo!("refactor required")
    }

    fn cpu_ptr(&self) -> NonNull<c_void> {
        todo!("refactor required")
    }

    fn gpu_ptr(&self) -> usize {
        todo!("refactor required")
    }

    fn length(&self) -> usize {
        self.buffer.size() as usize
    }
}

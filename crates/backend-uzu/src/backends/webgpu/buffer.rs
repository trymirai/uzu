use std::{os::raw::c_void, ptr::NonNull};

use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::{common::Buffer, webgpu::WebGPU};

#[derive(Debug)]
pub struct WebGPUBuffer {
    pub buffer: wgpu::Buffer,
}

impl WebGPUBuffer {
    fn as_native(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        let hal_buf = unsafe { self.buffer.as_hal::<wgpu::hal::api::Metal>() }.unwrap();
        unsafe { std::mem::transmute::<_, &Retained<ProtocolObject<dyn MTLBuffer>>>(&*hal_buf) } // TODO: very cringe
    }
}

impl Buffer for WebGPUBuffer {
    type Backend = WebGPU;

    fn set_label(
        &mut self,
        _label: Option<&str>,
    ) {
        // TODO: refactor into label on alloc
    }

    fn cpu_ptr(&self) -> NonNull<c_void> {
        // TODO: refactor with a new buffer infra
        self.as_native().contents()
    }

    fn gpu_ptr(&self) -> usize {
        // TODO: refactor with a new buffer infra
        self.as_native().gpu_address() as usize
    }

    fn length(&self) -> usize {
        self.buffer.size() as usize
    }
}

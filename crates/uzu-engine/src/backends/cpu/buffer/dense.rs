use std::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error},
    os::raw::c_void,
    ptr::NonNull,
};

use crate::backends::{
    common::{Buffer, CpuBuffer, GlobalBufferCpuGpu, GlobalGpuBuffer, GpuBuffer},
    cpu::Cpu,
};

#[derive(Debug)]
pub struct CpuDenseBuffer {
    ptr: NonNull<c_void>,
    layout: Layout,
}

impl CpuDenseBuffer {
    pub(in crate::backends::cpu) fn new(size: usize) -> Self {
        assert!(size > 0, "CPU buffer size must be nonzero");
        let layout = Layout::from_size_align(size, 64).expect("invalid CPU buffer layout");
        let ptr = NonNull::new(unsafe { alloc(layout) }).unwrap_or_else(|| handle_alloc_error(layout)).cast();
        Self {
            ptr,
            layout,
        }
    }
}

// The allocation has a stable address; callers synchronize access through buffer borrows and command buffers.
unsafe impl Send for CpuDenseBuffer {}
unsafe impl Sync for CpuDenseBuffer {}

impl Drop for CpuDenseBuffer {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) };
    }
}

impl Buffer for CpuDenseBuffer {
    type Backend = Cpu;

    fn size(&self) -> usize {
        self.layout.size()
    }
}

impl GpuBuffer for CpuDenseBuffer {}

impl CpuBuffer for CpuDenseBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.ptr
    }
}

impl GlobalGpuBuffer for CpuDenseBuffer {}

impl GlobalBufferCpuGpu for CpuDenseBuffer {
    fn handoff(self) -> Self {
        self
    }
}

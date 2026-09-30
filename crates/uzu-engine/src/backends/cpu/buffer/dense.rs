use std::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error},
    os::raw::c_void,
    ptr::NonNull,
};

use crate::backends::{
    common::{Buffer, BufferCpuAccessible, GlobalBuffer},
    cpu::Cpu,
};

#[derive(Debug)]
pub struct CpuBuffer {
    ptr: NonNull<c_void>,
    layout: Layout,
}

impl CpuBuffer {
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
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) };
    }
}

impl Buffer for CpuBuffer {
    type Backend = Cpu;

    fn size(&self) -> usize {
        self.layout.size()
    }
}

impl BufferCpuAccessible for CpuBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.ptr
    }
}

impl GlobalBuffer for CpuBuffer {}

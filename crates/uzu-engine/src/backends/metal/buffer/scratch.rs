use std::{os::raw::c_void, ptr::NonNull};

use crate::backends::{
    common::{Backend, Buffer, BufferCpuAccessible, ScratchBuffer, allocator::pool::PoolAllocation},
    metal::Metal,
};

impl Buffer for PoolAllocation<<Metal as Backend>::GlobalBuffer> {
    type Backend = Metal;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl BufferCpuAccessible for PoolAllocation<<Metal as Backend>::GlobalBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { self.page().cpu_ptr().byte_add(self.range().start) }
    }
}

impl ScratchBuffer for PoolAllocation<<Metal as Backend>::GlobalBuffer> {}

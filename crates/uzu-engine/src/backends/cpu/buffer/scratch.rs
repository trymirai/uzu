use std::{os::raw::c_void, ptr::NonNull};

use crate::backends::{
    common::{Backend, Buffer, BufferCpuAccessible, ScratchBuffer, allocator::pool::PoolAllocation},
    cpu::Cpu,
};

impl Buffer for PoolAllocation<<Cpu as Backend>::GlobalBuffer> {
    type Backend = Cpu;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl BufferCpuAccessible for PoolAllocation<<Cpu as Backend>::GlobalBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { self.page().cpu_ptr().byte_add(self.range().start) }
    }
}

impl ScratchBuffer for PoolAllocation<<Cpu as Backend>::GlobalBuffer> {}

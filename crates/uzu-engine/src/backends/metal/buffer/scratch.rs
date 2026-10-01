use std::{fmt::Debug, os::raw::c_void, ptr::NonNull};

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

impl Debug for PoolAllocation<<Metal as Backend>::GlobalBuffer> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("PoolAllocation<Metal::GlobalBuffer>")
            .field("page", &self.page())
            .field("range", &self.range())
            .finish_non_exhaustive()
    }
}

use std::{fmt::Debug, os::raw::c_void, ptr::NonNull};

use metal::MTLBuffer;

use crate::backends::{
    common::{Buffer, BufferCpuAccessible, GlobalBuffer, allocator::block::BlockAllocation},
    metal::{Metal, buffer::dense::MetalDenseBuffer},
};

impl Buffer for BlockAllocation<MetalDenseBuffer> {
    type Backend = Metal;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl BufferCpuAccessible for BlockAllocation<MetalDenseBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { self.page().mtl_buffer().contents().byte_add(self.range().start) }
    }
}

impl GlobalBuffer for BlockAllocation<MetalDenseBuffer> {}

impl Debug for BlockAllocation<MetalDenseBuffer> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("BlockAllocation<MetalDenseBuffer>")
            .field("page", &self.page())
            .field("range", &self.range())
            .finish_non_exhaustive()
    }
}

use std::{os::raw::c_void, ptr::NonNull};

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

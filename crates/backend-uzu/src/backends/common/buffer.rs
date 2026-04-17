use std::{fmt::Debug, ops::Range, os::raw::c_void, ptr::NonNull};

use super::Backend;

pub trait Buffer: Debug {
    type Backend: Backend<Buffer = Self>;

    fn set_label(
        &mut self,
        label: Option<&str>,
    );

    fn cpu_ptr(&self) -> NonNull<c_void>;

    fn gpu_ptr(&self) -> usize;

    fn length(&self) -> usize;
}

pub trait BufferGpuAddressRangeExt: Buffer {
    fn gpu_address_range(&self) -> Range<usize> {
        self.gpu_ptr()..(self.gpu_ptr() + self.length())
    }

    fn gpu_address_subrange(
        &self,
        subrange: Range<usize>,
    ) -> Range<usize> {
        assert!(subrange.end <= self.length(), "subrange overflow: subrange={:?} length={}", subrange, self.length());

        (self.gpu_ptr() + subrange.start)..(self.gpu_ptr() + subrange.end)
    }
}

impl<B: Buffer> BufferGpuAddressRangeExt for B {}

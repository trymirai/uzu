use std::{fmt::Debug, ops::Range};

use crate::backends::common::Backend;

pub trait Buffer: Debug {
    type Backend: Backend;

    fn gpu_ptr(&self) -> usize;

    fn size(&self) -> usize;

    fn set_label(
        &mut self,
        label: Option<&str>,
    );
}

pub trait BufferGpuAddressRangeExt: Buffer {
    fn gpu_address_range(&self) -> Range<usize> {
        self.gpu_ptr()..(self.gpu_ptr() + self.size())
    }

    fn gpu_address_subrange(
        &self,
        subrange: Range<usize>,
    ) -> Range<usize> {
        assert!(subrange.end <= self.size(), "subrange overflow: subrange={:?} length={}", subrange, self.size());

        (self.gpu_ptr() + subrange.start)..(self.gpu_ptr() + subrange.end)
    }
}

impl<B: Buffer> BufferGpuAddressRangeExt for B {}

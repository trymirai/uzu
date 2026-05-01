use std::{fmt::Debug, ops::Range};

use backend_uzu::backends::common::DenseBuffer;

pub trait Buffer: Debug {
    fn gpu_ptr(&self) -> usize;

    fn length(&self) -> usize;

    fn set_label(
        &mut self,
        label: Option<&str>,
    );
}

pub trait BufferGpuAddressRangeExt: DenseBuffer {
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

impl<B: DenseBuffer> BufferGpuAddressRangeExt for B {}

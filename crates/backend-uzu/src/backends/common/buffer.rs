use std::{fmt::Debug, ops::Range};

use bytesize::ByteSize;

use crate::backends::common::Backend;

pub trait Buffer: Debug {
    type Backend: Backend;

    fn gpu_ptr(&self) -> usize;

    fn size(&self) -> ByteSize;

    fn set_label(
        &mut self,
        label: Option<&str>,
    );
}

pub trait BufferGpuAddressRangeExt: Buffer {
    fn gpu_address_range(&self) -> Range<usize> {
        let len = self.size().as_u64() as usize;
        self.gpu_ptr()..(self.gpu_ptr() + len)
    }

    fn gpu_address_subrange(
        &self,
        subrange: Range<usize>,
    ) -> Range<usize> {
        let len = self.size().as_u64() as usize;
        assert!(subrange.end <= len, "subrange overflow: subrange={:?} length={}", subrange, len);

        (self.gpu_ptr() + subrange.start)..(self.gpu_ptr() + subrange.end)
    }
}

impl<B: Buffer> BufferGpuAddressRangeExt for B {}

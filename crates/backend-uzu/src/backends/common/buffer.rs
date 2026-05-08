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

// NOTE: we treat the buffer as the entire buffer but we also give a "safe" way to get the ref to the whole buffer from a range, this is a fundamental limitation of the current design, but we can nicely fix this after migration to allocations

pub struct BufferRangeRef<'a, B: Buffer> {
    buffer: &'a B,
    range: Range<usize>,
}

impl<'a, B: Buffer> BufferRangeRef<'a, B> {
    pub(super) fn new(
        buffer: &'a B,
        range: Range<usize>,
    ) -> Self {
        Self {
            buffer,
            range,
        }
    }

    pub fn buffer(&self) -> &'a B {
        self.buffer
    }

    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

pub trait AsBufferRangeRef {
    type Buffer: Buffer;

    fn as_buffer_range_ref<'a>(&'a self) -> BufferRangeRef<'a, Self::Buffer>;
}

pub struct BufferRangeMut<'a, B: Buffer> {
    buffer: &'a B,
    range: Range<usize>,
}

impl<'a, B: Buffer> BufferRangeMut<'a, B> {
    pub(super) fn new_exclusive(
        buffer: &'a mut B,
        range: Range<usize>,
    ) -> Self {
        Self {
            buffer,
            range,
        }
    }

    pub(super) unsafe fn new_shared(
        buffer: &'a B,
        range: Range<usize>,
    ) -> Self {
        Self {
            buffer,
            range,
        }
    }

    pub fn buffer(&self) -> &'a B {
        self.buffer
    }

    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

pub trait AsBufferRangeMut: AsBufferRangeRef {
    fn as_buffer_range_mut<'a>(&'a mut self) -> BufferRangeMut<'a, Self::Buffer>;
}

impl<B: Buffer> AsBufferRangeRef for B {
    type Buffer = B;

    fn as_buffer_range_ref<'a>(&'a self) -> BufferRangeRef<'a, B> {
        BufferRangeRef::new(self, 0..self.size())
    }
}

impl<B: Buffer> AsBufferRangeMut for B {
    fn as_buffer_range_mut<'a>(&'a mut self) -> BufferRangeMut<'a, B> {
        let size = self.size();

        BufferRangeMut::new_exclusive(self, 0..size)
    }
}

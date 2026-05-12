// NOTE: we treat the buffer as the entire buffer,
// but we also give a "safe" way to get the ref to the whole buffer from a range,
// this is a fundamental limitation of the current design,
// but we can nicely fix this after migration to allocations

use std::ops::Range;

use crate::backends::common::Buffer;

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

    pub(crate) fn subrange(
        self,
        range: Range<usize>,
    ) -> Self {
        assert!(range.end <= self.range.len(), "buffer subrange exceeds range");
        Self {
            buffer: self.buffer,
            range: self.range.start + range.start..self.range.start + range.end,
        }
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

    pub(crate) fn subrange(
        self,
        range: Range<usize>,
    ) -> Self {
        assert!(range.end <= self.range.len(), "buffer subrange exceeds range");
        Self {
            buffer: self.buffer,
            range: self.range.start + range.start..self.range.start + range.end,
        }
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

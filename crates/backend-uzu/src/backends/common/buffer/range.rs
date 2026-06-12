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
    pub fn new(
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

    pub fn subrange(
        self,
        range: Range<usize>,
    ) -> Self {
        Self {
            buffer: self.buffer,
            range: subrange(self.range, range),
        }
    }
}

pub trait AsBufferRangeRef {
    type Buffer: Buffer;

    fn as_buffer_range_ref(&self) -> BufferRangeRef<'_, Self::Buffer>;
}

pub struct BufferRangeMut<'a, B: Buffer> {
    buffer: &'a B,
    range: Range<usize>,
}

impl<'a, B: Buffer> BufferRangeMut<'a, B> {
    pub fn new_exclusive(
        buffer: &'a mut B,
        range: Range<usize>,
    ) -> Self {
        Self {
            buffer,
            range,
        }
    }

    pub unsafe fn new_shared(
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

    pub fn subrange(
        self,
        range: Range<usize>,
    ) -> Self {
        Self {
            buffer: self.buffer,
            range: subrange(self.range, range),
        }
    }
}

pub trait AsBufferRangeMut: AsBufferRangeRef {
    fn as_buffer_range_mut(&mut self) -> BufferRangeMut<'_, Self::Buffer>;
}

impl<B: Buffer> AsBufferRangeRef for B {
    type Buffer = B;

    fn as_buffer_range_ref(&self) -> BufferRangeRef<'_, B> {
        BufferRangeRef::new(self, 0..self.size())
    }
}

impl<B: Buffer> AsBufferRangeMut for B {
    fn as_buffer_range_mut(&mut self) -> BufferRangeMut<'_, B> {
        BufferRangeMut::new_exclusive(self, 0..self.size())
    }
}

fn subrange(
    buffer_range: Range<usize>,
    subrange: Range<usize>,
) -> Range<usize> {
    assert!(subrange.end <= buffer_range.len(), "buffer subrange exceeds range");
    buffer_range.start + subrange.start..buffer_range.start + subrange.end
}

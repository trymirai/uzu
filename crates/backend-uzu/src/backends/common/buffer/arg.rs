use crate::backends::common::{AsBufferRangeMut, AsBufferRangeRef, Backend, Buffer, BufferRangeMut};

// TODO: This whole thing is horrible and should be unified with AsBufferRangeRef
pub trait BufferArg<'a, B: Backend>: Copy {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = B>, usize, usize);
}

impl<'a, B: Backend, T: AsBufferRangeRef<Buffer: Buffer<Backend = B>>> BufferArg<'a, B> for &'a T {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = B>, usize, usize) {
        let buffer_range = self.as_buffer_range_ref();
        let (buffer, range) = (buffer_range.buffer(), buffer_range.range());
        (buffer, range.start, range.end - range.start)
    }
}

impl<'a, B: Backend> BufferArg<'a, B> for &'a dyn Buffer<Backend = B> {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = B>, usize, usize) {
        (self, 0, self.size())
    }
}

impl<'a, B: Backend, T: BufferArg<'a, B>> BufferArg<'a, B> for (T, usize) {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = B>, usize, usize) {
        let (buffer, offset, length) = self.0.into_parts();
        (buffer, offset + self.1, length - self.1)
    }
}

pub trait BufferArgMut<'a, B: Backend> {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = B>, usize, usize);
}

impl<'a, B: Backend, Buf: Buffer<Backend = B>> BufferArgMut<'a, B> for BufferRangeMut<'a, Buf> {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = B>, usize, usize) {
        let (buffer, range) = (self.buffer(), self.range());
        (buffer, range.start, range.len())
    }
}

impl<'a, B: Backend, T: AsBufferRangeMut<Buffer: Buffer<Backend = B>>> BufferArgMut<'a, B> for &'a mut T {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = B>, usize, usize) {
        let buffer_range = self.as_buffer_range_mut();
        let (buffer, range) = (buffer_range.buffer(), buffer_range.range());
        (buffer, range.start, range.end - range.start)
    }
}

impl<'a, B: Backend> BufferArgMut<'a, B> for &'a mut dyn Buffer<Backend = B> {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = B>, usize, usize) {
        (self, 0, self.size())
    }
}

impl<'a, B: Backend, T: BufferArgMut<'a, B>> BufferArgMut<'a, B> for (T, usize) {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = B>, usize, usize) {
        let (buffer, offset, length) = self.0.into_parts();
        (buffer, offset + self.1, length - self.1)
    }
}

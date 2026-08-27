use std::{
    ops::{Bound, RangeBounds},
    range::Range,
};

use bytemuck::{AnyBitPattern, NoUninit};

use crate::backends::common::{Backend, Buffer, BufferCpuAccessible};

// TODO: revisit as_slice/copy bounds and remove horrible unsafe metal kernel argument copyin hack
// TODO: also I'm not a big fan of this implementation. The implementation can be clean when BufferRef is on T
// TODO: but then &(impl BufferRef + ?Sized) everywhere it's used is much worse than a slightly messy trait system

pub trait BufferRef: Copy {
    type Backend: Backend;
    type Buffer: Buffer<Backend = Self::Backend> + ?Sized;

    fn size(&self) -> usize;

    fn parts<'a>(self) -> (&'a Self::Buffer, Range<usize>)
    where
        Self: 'a;

    fn as_slice<'a, T: AnyBitPattern>(self) -> &'a [T]
    where
        Self::Buffer: BufferCpuAccessible,
        Self: 'a,
    {
        let (buffer, range) = self.parts();
        let bytes = unsafe {
            std::slice::from_raw_parts(buffer.cpu_ptr().as_ptr().cast::<u8>().add(range.start), range.iter().len())
        };
        bytemuck::cast_slice(bytes)
    }

    fn copyout<T: AnyBitPattern>(self) -> Vec<T>
    where
        Self::Buffer: BufferCpuAccessible,
    {
        self.as_slice().to_vec()
    }

    fn subrange<'a>(
        self,
        range: impl RangeBounds<usize>,
    ) -> Subbuffer<'a, Self::Buffer, false>
    where
        Self: 'a,
    {
        let (buffer, buffer_range) = self.parts();
        Subbuffer {
            buffer,
            range: subrange(buffer_range, range),
        }
    }
}

pub trait BufferMut: Sized {
    type Backend: Backend;
    type Buffer: Buffer<Backend = Self::Backend> + ?Sized;

    fn size(&self) -> usize;

    fn parts<'a>(self) -> (&'a Self::Buffer, Range<usize>)
    where
        Self: 'a;

    fn reborrow(&mut self) -> impl BufferMut<Backend = Self::Backend, Buffer = Self::Buffer>;

    fn as_ref(&self) -> impl BufferRef<Backend = Self::Backend, Buffer = Self::Buffer>;

    #[expect(clippy::wrong_self_convention)]
    fn as_slice_mut<'a, T: NoUninit + AnyBitPattern>(self) -> &'a mut [T]
    where
        Self::Buffer: BufferCpuAccessible,
        Self: 'a,
    {
        let (buffer, range) = self.parts();
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(buffer.cpu_ptr().as_ptr().cast::<u8>().add(range.start), range.iter().len())
        };
        bytemuck::cast_slice_mut(bytes)
    }

    fn copyin(
        self,
        data: &[impl NoUninit + AnyBitPattern],
    ) where
        Self::Buffer: BufferCpuAccessible,
    {
        self.as_slice_mut().copy_from_slice(data);
    }

    fn subrange_mut<'a>(
        self,
        range: impl RangeBounds<usize>,
    ) -> Subbuffer<'a, Self::Buffer, true>
    where
        Self: 'a,
    {
        let (buffer, buffer_range) = self.parts();
        Subbuffer {
            buffer,
            range: subrange(buffer_range, range),
        }
    }

    fn split_at<'a>(
        self,
        byte_offset: usize,
    ) -> (Subbuffer<'a, Self::Buffer, true>, Subbuffer<'a, Self::Buffer, true>)
    where
        Self: 'a,
    {
        let (buffer, range) = self.parts();
        assert!(byte_offset <= range.iter().len(), "split offset exceeds buffer range size");
        let split = range.start + byte_offset;
        (
            Subbuffer {
                buffer,
                range: (range.start..split).into(),
            },
            Subbuffer {
                buffer,
                range: (split..range.end).into(),
            },
        )
    }
}

pub struct Subbuffer<'a, B: Buffer + ?Sized, const MUTABLE: bool> {
    buffer: &'a B,
    range: Range<usize>,
}

impl<B: Buffer + ?Sized> Clone for Subbuffer<'_, B, false> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: Buffer + ?Sized> Copy for Subbuffer<'_, B, false> {}

impl<B: Buffer + ?Sized> BufferRef for Subbuffer<'_, B, false> {
    type Backend = B::Backend;
    type Buffer = B;

    fn size(&self) -> usize {
        self.range.iter().len()
    }

    fn parts<'a>(self) -> (&'a B, Range<usize>)
    where
        Self: 'a,
    {
        (self.buffer, self.range)
    }
}

impl<B: Buffer + ?Sized> BufferMut for Subbuffer<'_, B, true> {
    type Backend = B::Backend;
    type Buffer = B;

    fn size(&self) -> usize {
        self.range.iter().len()
    }

    fn parts<'a>(self) -> (&'a B, Range<usize>)
    where
        Self: 'a,
    {
        (self.buffer, self.range)
    }

    fn reborrow(&mut self) -> impl BufferMut<Backend = Self::Backend, Buffer = B> {
        Subbuffer::<_, true> {
            buffer: self.buffer,
            range: self.range,
        }
    }

    fn as_ref(&self) -> impl BufferRef<Backend = Self::Backend, Buffer = B> {
        Subbuffer::<_, false> {
            buffer: self.buffer,
            range: self.range,
        }
    }
}

pub fn subrange(
    buffer_range: Range<usize>,
    subrange: impl RangeBounds<usize>,
) -> Range<usize> {
    let start = match subrange.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1).expect("subrange start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match subrange.end_bound() {
        Bound::Included(&end) => end.checked_add(1).expect("subrange end overflow"),
        Bound::Excluded(&end) => end,
        Bound::Unbounded => buffer_range.iter().len(),
    };
    assert!(start <= end && end <= buffer_range.iter().len(), "invalid subrange");
    (buffer_range.start + start..buffer_range.start + end).into()
}

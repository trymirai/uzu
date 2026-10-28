use std::{any::Any, os::raw::c_void, ptr::NonNull, range::Range};

use bytemuck::{AnyBitPattern, NoUninit};

use crate::backends::common::{Backend, BufferMut, BufferRef};

pub mod global;
pub mod sparse;

pub mod reference;

pub trait Buffer: Any + Send + Sync + Unpin {
    type Backend: Backend;

    fn size(&self) -> usize;
}

pub trait GpuBuffer: Buffer {}

pub trait CpuBuffer: Buffer {
    fn cpu_ptr(&self) -> NonNull<c_void>;

    fn as_slice<T: AnyBitPattern>(&self) -> &[T] {
        let bytes = unsafe { std::slice::from_raw_parts(self.cpu_ptr().as_ptr().cast::<u8>(), self.size()) };
        bytemuck::cast_slice(bytes)
    }

    fn as_slice_mut<T: NoUninit + AnyBitPattern>(&mut self) -> &mut [T] {
        let bytes = unsafe { std::slice::from_raw_parts_mut(self.cpu_ptr().as_ptr().cast::<u8>(), self.size()) };
        bytemuck::cast_slice_mut(bytes)
    }

    fn copyin(
        &mut self,
        data: &[impl NoUninit + AnyBitPattern],
    ) {
        self.as_slice_mut().copy_from_slice(data);
    }

    fn copyout<T: AnyBitPattern>(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }
}

impl<B: GpuBuffer + ?Sized> BufferRef for &B {
    type Backend = B::Backend;
    type Buffer = B;

    fn size(&self) -> usize {
        Buffer::size(*self)
    }

    fn parts<'a>(self) -> (&'a B, Range<usize>)
    where
        Self: 'a,
    {
        (self, (0..self.size()).into())
    }
}

impl<B: GpuBuffer + ?Sized> BufferMut for &mut B {
    type Backend = B::Backend;
    type Buffer = B;

    fn size(&self) -> usize {
        Buffer::size(&**self)
    }

    fn parts<'a>(self) -> (&'a B, Range<usize>)
    where
        Self: 'a,
    {
        (self, (0..self.size()).into())
    }

    fn reborrow(&mut self) -> impl BufferMut<Backend = Self::Backend, Buffer = B> {
        &mut **self
    }

    fn as_ref(&self) -> impl BufferRef<Backend = Self::Backend, Buffer = B> {
        &**self
    }
}

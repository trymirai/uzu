use std::{any::Any, fmt::Debug, range::Range};

use crate::backends::common::{Backend, BufferMut, BufferRef};

pub mod constant;
pub mod cpu_accessible;
pub mod global;
pub mod scratch;
pub mod sparse;

pub mod reference;

pub trait Buffer: Any + Debug + Send + Sync + Unpin {
    type Backend: Backend;

    fn size(&self) -> usize;
}

impl<B: Buffer + ?Sized> BufferRef for &B {
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

impl<B: Buffer + ?Sized> BufferMut for &mut B {
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

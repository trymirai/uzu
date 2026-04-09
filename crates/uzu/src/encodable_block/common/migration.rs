use std::{cell::RefCell, ops::Range, rc::Rc};

use crate::{
    array::Array,
    backends::common::{
        Allocation, Backend,
        kernel::{BufferArg, BufferArgMut},
    },
};

// Temporary bridge for the encodable-block migration. Migrated blocks expose allocation-shaped
// inputs, but some callers still source buffers from ForwardPassState arrays. Remove the borrowed
// path once those callers become allocator-native.
pub(crate) enum BlockAllocation<B: Backend> {
    Owned(Allocation<B>),
    Borrowed {
        buffer_owner: Rc<RefCell<B::Buffer>>,
        range: Range<usize>,
    },
}

impl<B: Backend> BlockAllocation<B> {
    pub(crate) fn from_array(array: &Array<B>) -> Self {
        Self::Borrowed {
            buffer_owner: array.buffer(),
            range: array.offset()..array.offset() + array.size(),
        }
    }

    pub(crate) fn as_buffer_range(&self) -> (&B::Buffer, Range<usize>) {
        match self {
            Self::Owned(allocation) => allocation.as_buffer_range(),
            Self::Borrowed {
                buffer_owner,
                range,
            } => (unsafe { &*buffer_owner.as_ptr() }, range.clone()),
        }
    }
}

impl<B: Backend> From<Allocation<B>> for BlockAllocation<B> {
    fn from(value: Allocation<B>) -> Self {
        Self::Owned(value)
    }
}

impl<'a, B: Backend> BufferArg<'a, B::Buffer> for &'a BlockAllocation<B> {
    fn into_parts(self) -> (&'a B::Buffer, usize) {
        let (buffer, range) = self.as_buffer_range();
        (buffer, range.start)
    }
}

impl<'a, B: Backend> BufferArgMut<'a, B::Buffer> for &'a mut BlockAllocation<B> {
    fn into_parts(self) -> (&'a B::Buffer, usize) {
        let (buffer, range) = self.as_buffer_range();
        (buffer, range.start)
    }
}

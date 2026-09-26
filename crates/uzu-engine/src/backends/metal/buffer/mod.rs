use std::range::Range;

use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::{
    backends::{
        common::{Backend, Buffer},
        metal::{Metal, buffer::sparse::MetalSparseBuffer},
    },
    utils::downcast::downcast_ref,
};

pub mod dense;
pub mod sparse;

pub(super) trait MetalBufferExt: Buffer<Backend = Metal> {
    fn downcast(&self) -> (&Retained<ProtocolObject<dyn MTLBuffer>>, usize) {
        if let Some(buffer) = downcast_ref::<<Metal as Backend>::GlobalBuffer>(self) {
            (buffer.buffer().mtl_buffer(), buffer.offset())
        } else if let Some(buffer) = downcast_ref::<MetalSparseBuffer>(self) {
            (buffer.mtl_buffer(), 0)
        } else {
            unreachable!()
        }
    }

    fn gpu_address(&self) -> u64 {
        let (buffer, offset) = self.downcast();
        buffer.gpu_address() + offset as u64
    }

    fn gpu_address_subrange(
        &self,
        subrange: Range<usize>,
    ) -> Range<u64> {
        assert!(subrange.end <= self.size(), "subrange overflow: subrange={:?} length={}", subrange, self.size());
        let buffer_pointer = self.gpu_address();
        ((buffer_pointer + subrange.start as u64)..(buffer_pointer + subrange.end as u64)).into()
    }
}

impl<T: Buffer<Backend = Metal> + ?Sized> MetalBufferExt for T {}

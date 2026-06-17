use std::{any::Any, fmt::Debug, ops::Range};

use crate::backends::common::{Backend, hazard_tracker::ResourceHandle};

pub mod dense;
pub mod range;
pub mod sparse;

pub trait Buffer: Any + Debug {
    type Backend: Backend;

    fn gpu_ptr(&self) -> usize;

    fn size(&self) -> usize;

    /// Opaque handle to the underlying backend resource, used to scope memory barriers to the
    /// exact resources involved. Defaults to `None` (backends without a notion of resource
    /// handles, e.g. CPU, fall back to a coarse barrier).
    fn resource_handle(&self) -> ResourceHandle {
        None
    }
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

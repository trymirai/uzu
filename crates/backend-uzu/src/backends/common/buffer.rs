use std::{any::Any, fmt::Debug, ops::Range};

use crate::backends::common::Backend;

pub trait Buffer: Any + Debug {
    type Backend: Backend;

    fn as_bytes_slice_range(
        &self,
        context: Option<&<Self::Backend as Backend>::Context>,
        range: Range<usize>,
    ) -> Result<&[u8], <Self::Backend as Backend>::Error>;

    fn gpu_ptr(&self) -> usize;

    fn size(&self) -> usize;
}

pub trait BufferGpuAddressRangeExt: Buffer {
    fn gpu_address_subrange(
        &self,
        subrange: Range<usize>,
    ) -> Range<usize> {
        assert!(subrange.end <= self.size(), "subrange overflow: subrange={:?} length={}", subrange, self.size());

        (self.gpu_ptr() + subrange.start)..(self.gpu_ptr() + subrange.end)
    }
}

impl<B: Buffer> BufferGpuAddressRangeExt for B {}

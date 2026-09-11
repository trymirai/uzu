use std::ops::Range;

use crate::backends::common::{Backend, Buffer};

pub trait SparseBuffer: Buffer<Backend: Backend<SparseBuffer = Self>> {
    fn map(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn unmap(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn page_size_bytes(&self) -> usize;
}

pub trait SparseBufferExt: SparseBuffer {
    fn total_pages(&self) -> usize {
        self.size() / self.page_size_bytes()
    }
}

impl<B: SparseBuffer> SparseBufferExt for B {}

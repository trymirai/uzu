use std::range::Range;

use crate::backends::common::{Backend, Buffer};

pub trait SparseBuffer: Buffer<Backend: Backend<SparseBuffer = Self>> {
    fn map(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: impl Into<Range<usize>>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn unmap(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: impl Into<Range<usize>>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn page_size_bytes(&self) -> usize;

    fn total_pages(&self) -> usize {
        self.size() / self.page_size_bytes()
    }
}

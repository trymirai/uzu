use std::ops::Range;

use crate::backends::common::{Backend, Buffer};
pub trait SparseBuffer: Buffer<Backend: Backend<SparseBuffer = Self>> {
    fn mapping(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn unmapping(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}

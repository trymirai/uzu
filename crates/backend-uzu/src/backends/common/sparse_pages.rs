use std::{fmt::Debug, ops::Range};

use crate::backends::common::Backend;

pub struct SparsePagesOperation {
    pub map: bool,
    pub pages: Range<usize>,
}

pub trait SparsePages: Debug {
    type Backend: Backend;

    fn buffer(&self) -> &<Self::Backend as Backend>::Buffer;

    fn buffer_mut(&mut self) -> &mut <Self::Backend as Backend>::Buffer;

    fn execute(
        &mut self,
        operations: &[SparsePagesOperation],
    );

    fn page_size(&self) -> usize;

    fn total_pages(&self) -> usize;
}

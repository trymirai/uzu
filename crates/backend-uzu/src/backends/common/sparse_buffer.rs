use std::fmt::Debug;

use crate::backends::common::Backend;

pub trait SparseBuffer: Debug {
    type Backend: Backend<SparseBuffer = Self>;

    fn buffer(&self) -> &<Self::Backend as Backend>::Buffer;

    fn buffer_mut(&mut self) -> &mut <Self::Backend as Backend>::Buffer;

    fn capacity(&self) -> usize;

    fn extend(
        &mut self,
        add_length: usize,
    );

    fn length(&self) -> usize;
}

use crate::backends::common::{Backend, Buffer};

pub trait SparseBuffer: Buffer<Backend: Backend<SparseBuffer = Self>> {
    fn map(
        &mut self,
        until: usize,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}

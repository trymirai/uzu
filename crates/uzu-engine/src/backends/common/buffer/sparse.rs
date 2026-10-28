use crate::backends::common::{Backend, GpuBuffer};

pub trait SparseBuffer: GpuBuffer<Backend: Backend<SparseBuffer = Self>> {
    fn map(
        &mut self,
        until: usize,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}

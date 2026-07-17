use std::{path::Path, sync::Arc};

use crate::backends::common::{Allocation, AllocationPool, AllocationType, Backend, CommandBuffer};

pub trait Context: Sized + Send + Sync {
    type Backend: Backend<Context = Self>;

    fn new() -> Result<Arc<Self>, <Self::Backend as Backend>::Error>;

    fn create_command_buffer(
        &self
    ) -> Result<<<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Initial, <Self::Backend as Backend>::Error>;

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::DenseBuffer, <Self::Backend as Backend>::Error>;

    fn create_allocation(
        &self,
        size: usize,
        allocation_type: AllocationType<Self::Backend>,
    ) -> Result<Allocation<Self::Backend>, <Self::Backend as Backend>::Error>;

    fn create_allocation_pool(
        &self,
        reusable: bool,
    ) -> AllocationPool<Self::Backend>;

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error>;

    fn peak_memory_usage(&self) -> Option<usize>;

    fn enable_capture();

    fn start_capture(
        &self,
        trace_path: &Path,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn stop_capture(&self) -> Result<(), <Self::Backend as Backend>::Error>;

    fn sparse_buffers_supported(&self) -> bool;

    fn supports_mxu(&self) -> bool {
        false
    }
}

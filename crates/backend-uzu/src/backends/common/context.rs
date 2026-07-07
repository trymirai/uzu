use std::{path::Path, rc::Rc};

use crate::backends::common::{Allocation, AllocationPool, AllocationType, Backend, CommandBuffer};

pub trait Context: Sized {
    type Backend: Backend<Context = Self>;

    fn new() -> Result<Rc<Self>, <Self::Backend as Backend>::Error>;

    fn recommended_async_batch_size(
        &self,
        model_path: &Path,
    ) -> Result<usize, <Self::Backend as Backend>::Error>;

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

    /// Whether the GPU is Apple family 9 or newer (M3/A17 and up), which brings
    /// dynamic GPU-core caching. Used to gate the chunked GDN prefill path
    /// (family-9 GPUs run it profitably at large T even without an MXU).
    /// Defaults to `false` so CPU and pre-family-9 devices stay on the
    /// recurrent path.
    fn supports_dynamic_caching(&self) -> bool {
        false
    }
}

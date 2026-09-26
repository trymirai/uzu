use std::{path::Path, sync::Arc};

use crate::backends::common::{Backend, CommandBuffer, DeviceCapabilities};

pub trait Context: Sized + Send + Sync {
    type Backend: Backend<Context = Self>;

    fn new() -> Result<Arc<Self>, <Self::Backend as Backend>::Error>;

    fn device_name(&self) -> Option<&str>;

    fn create_command_buffer(
        &self,
        name: Option<&str>,
        allocation_pool: Option<Arc<<Self::Backend as Backend>::AllocationPool>>,
    ) -> Result<<<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding, <Self::Backend as Backend>::Error>;

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::GlobalBuffer, <Self::Backend as Backend>::Error>;

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error>;

    fn create_allocation_pool(&self) -> Arc<<Self::Backend as Backend>::AllocationPool>;

    fn peak_memory_usage(&self) -> Option<usize>;

    fn enable_capture();

    fn start_capture(
        &self,
        trace_path: &Path,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn stop_capture(&self) -> Result<(), <Self::Backend as Backend>::Error>;

    fn device_capabilities(&self) -> DeviceCapabilities;
}

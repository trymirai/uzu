use std::{ffi::c_void, path::Path, ptr::NonNull, sync::Arc};

use memmap2::MmapMut;

use crate::backends::common::{
    Allocation, AllocationPool, AllocationType, Backend, CommandBuffer, DeviceCapabilities, MappedFile,
};

pub trait Context: Sized + Send + Sync {
    type Backend: Backend<Context = Self>;

    fn new() -> Result<Arc<Self>, <Self::Backend as Backend>::Error>;

    fn create_command_buffer(
        &self,
        name: Option<&str>,
    ) -> Result<<<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Initial, <Self::Backend as Backend>::Error>;

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::DenseBuffer, <Self::Backend as Backend>::Error>;

    /// A buffer over `size` bytes of host memory at the page-aligned `pointer`, without copying.
    ///
    /// # Safety
    /// The memory must stay valid, and mapped, for the buffer's whole lifetime.
    unsafe fn create_buffer_over_host_memory(
        &self,
        pointer: NonNull<c_void>,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::DenseBuffer, <Self::Backend as Backend>::Error>;

    fn max_buffer_length(&self) -> usize;

    fn create_allocation(
        &self,
        size: usize,
        allocation_type: AllocationType<Self::Backend>,
    ) -> Result<Allocation<Self::Backend>, <Self::Backend as Backend>::Error>;

    /// A mapped file whose ranges become allocations used in place.
    fn map_file(
        &self,
        map: MmapMut,
    ) -> MappedFile<Self::Backend>;

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

    fn device_capabilities(&self) -> DeviceCapabilities;
}

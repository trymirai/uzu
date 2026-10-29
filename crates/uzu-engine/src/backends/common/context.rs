use std::{path::Path, sync::Arc};

use bytemuck::AnyBitPattern;

use crate::backends::common::{Backend, BufferRef, CommandBuffer, DeviceCapabilities};

pub trait Context: Sized + Send + Sync {
    type Backend: Backend<Context = Self>;

    fn new() -> impl Future<Output = Result<Arc<Self>, <Self::Backend as Backend>::Error>> + Send;

    fn device_name(&self) -> Option<&str>;

    fn create_command_buffer(
        &self,
        name: Option<&str>,
        allocation_pool: Option<Arc<<Self::Backend as Backend>::AllocationPool>>,
    ) -> Result<<<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding, <Self::Backend as Backend>::Error>;

    fn create_buffer_gpu(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::GlobalGpuBuffer, <Self::Backend as Backend>::Error>;

    fn create_buffer_cpu_gpu(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::GlobalBufferCpuGpu, <Self::Backend as Backend>::Error>;

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error>;

    fn copyout<T: AnyBitPattern>(
        &self,
        buffer: impl BufferRef<Backend = Self::Backend, Buffer = <Self::Backend as Backend>::GlobalGpuBuffer>,
    ) -> impl Future<Output = Result<Vec<T>, <Self::Backend as Backend>::Error>> + Send;

    fn create_allocation_pool(&self) -> Arc<<Self::Backend as Backend>::AllocationPool>;

    fn peak_memory_usage(&self) -> Option<usize>;

    fn start_capture(
        &self,
        output_path: &Path,
    ) -> Result<<Self::Backend as Backend>::CaptureSpan, <Self::Backend as Backend>::Error>;

    fn device_capabilities(&self) -> DeviceCapabilities;
}

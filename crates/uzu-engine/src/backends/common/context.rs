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

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::GlobalBuffer, <Self::Backend as Backend>::Error>;

    /// A global buffer that GPU work writes for the CPU to read in place once that work completes.
    fn create_readback_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::GlobalBuffer, <Self::Backend as Backend>::Error>;

    fn create_buffer_initializer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::GlobalBufferInitializer, <Self::Backend as Backend>::Error>;

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error>;

    /// Reads `buffer` once the work submitted before this call completes.
    fn copyout<T: AnyBitPattern>(
        &self,
        buffer: impl BufferRef<Backend = Self::Backend, Buffer = <Self::Backend as Backend>::GlobalBuffer>,
    ) -> impl Future<Output = Result<Vec<T>, <Self::Backend as Backend>::Error>> + Send;

    /// Reads `buffer` after the command buffers writing it completed: in place where global buffers are CPU-accessible,
    /// otherwise through `copyout`.
    fn read_completed<T: AnyBitPattern>(
        &self,
        buffer: impl BufferRef<Backend = Self::Backend, Buffer = <Self::Backend as Backend>::GlobalBuffer>,
    ) -> impl Future<Output = Result<Vec<T>, <Self::Backend as Backend>::Error>> + Send {
        async move {
            if self.device_capabilities().contains(DeviceCapabilities::GLOBAL_BUFFERS_CPU_ACCESSIBLE) {
                Ok(buffer.copyout())
            } else {
                self.copyout(buffer).await
            }
        }
    }

    fn create_allocation_pool(&self) -> Arc<<Self::Backend as Backend>::AllocationPool>;

    fn peak_memory_usage(&self) -> Option<usize>;

    fn start_capture(
        &self,
        output_path: &Path,
    ) -> Result<<Self::Backend as Backend>::CaptureSpan, <Self::Backend as Backend>::Error>;

    fn device_capabilities(&self) -> DeviceCapabilities;
}

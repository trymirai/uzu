use std::{
    convert::Infallible,
    path::Path,
    sync::{Arc, Weak, mpsc},
    thread,
};

use crate::backends::{
    common::{
        Backend, BufferRef, Context, CpuBuffer, DeviceCapabilities,
        allocator::{bump::BumpAllocator, pool::PoolAllocator},
    },
    cpu::{Cpu, buffer::dense::CpuDenseBuffer, command_buffer::CpuCommandBufferEncoding, error::CpuError},
};

pub struct CpuContext {
    pub(super) command_queue: mpsc::Sender<Box<dyn FnOnce() + Send>>,
    weak_self: Weak<CpuContext>,
}

impl Context for CpuContext {
    type Backend = Cpu;

    async fn new() -> Result<Arc<Self>, CpuError> {
        let (command_queue_sender, command_queue_receiever) = mpsc::channel::<Box<dyn FnOnce() + Send>>();

        thread::spawn(|| {
            for command_buffer in command_queue_receiever {
                command_buffer();
            }
        });

        Ok(Arc::new_cyclic(|weak_self| CpuContext {
            command_queue: command_queue_sender,
            weak_self: weak_self.clone(),
        }))
    }

    fn device_name(&self) -> Option<&str> {
        None
    }

    fn create_command_buffer(
        &self,
        _name: Option<&str>,
        allocation_pool: Option<Arc<<Cpu as Backend>::AllocationPool>>,
    ) -> Result<CpuCommandBufferEncoding, CpuError> {
        let constant_allocator = BumpAllocator::new(256 * 1024, 1);
        let allocation_pool = allocation_pool.unwrap_or_else(|| self.create_allocation_pool());
        Ok(CpuCommandBufferEncoding::new(self.weak_self.upgrade().unwrap(), constant_allocator, allocation_pool))
    }

    fn create_buffer_gpu(
        &self,
        size: usize,
    ) -> Result<<Cpu as Backend>::GlobalGpuBuffer, CpuError> {
        Ok(CpuDenseBuffer::new(size))
    }

    fn create_buffer_cpu_gpu(
        &self,
        size: usize,
    ) -> Result<<Cpu as Backend>::GlobalBufferCpuGpu, CpuError> {
        Ok(CpuDenseBuffer::new(size))
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        Err(CpuError::NotSupported)
    }

    async fn copyout<T: bytemuck::AnyBitPattern>(
        &self,
        buffer: impl BufferRef<Backend = Cpu, Buffer = CpuDenseBuffer>,
    ) -> Result<Vec<T>, CpuError> {
        let (buffer, range) = buffer.parts();
        let bytes = unsafe {
            std::slice::from_raw_parts(buffer.cpu_ptr().as_ptr().cast::<u8>().add(range.start), range.iter().len())
        };
        Ok(bytemuck::cast_slice(bytes).to_vec())
    }

    fn create_allocation_pool(&self) -> Arc<<Cpu as Backend>::AllocationPool> {
        PoolAllocator::new()
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }

    fn start_capture(
        &self,
        _output_path: &Path,
    ) -> Result<Infallible, CpuError> {
        Err(CpuError::NotSupported)
    }

    fn device_capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::GLOBAL_BUFFERS_CPU_ACCESSIBLE
    }
}

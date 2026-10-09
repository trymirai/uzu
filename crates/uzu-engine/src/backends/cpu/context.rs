use std::{
    path::Path,
    sync::{Arc, Weak, mpsc},
    thread,
};

use crate::backends::{
    common::{
        Backend, Context, DeviceCapabilities,
        allocator::{bump::BumpAllocator, pool::PoolAllocator},
    },
    cpu::{Cpu, buffer::dense::CpuBuffer, command_buffer::CpuCommandBufferEncoding, error::CpuError},
};

pub struct CpuContext {
    pub(super) command_queue: mpsc::Sender<Box<dyn FnOnce() + Send>>,
    weak_self: Weak<CpuContext>,
}

impl Context for CpuContext {
    type Backend = Cpu;

    fn new() -> Result<Arc<Self>, CpuError> {
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
        timestamps: bool,
    ) -> Result<CpuCommandBufferEncoding, CpuError> {
        let constant_allocator = BumpAllocator::new(256 * 1024);
        let allocation_pool = allocation_pool.unwrap_or_else(|| self.create_allocation_pool());
        Ok(CpuCommandBufferEncoding::new(
            constant_allocator,
            allocation_pool,
            self.weak_self.upgrade().unwrap(),
            timestamps,
        ))
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Cpu as Backend>::GlobalBuffer, CpuError> {
        Ok(CpuBuffer::new(size))
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        Err(CpuError::NotSupported)
    }

    fn create_allocation_pool(&self) -> Arc<<Cpu as Backend>::AllocationPool> {
        PoolAllocator::new()
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }

    fn enable_capture() {}

    fn start_capture(
        &self,
        _trace_path: &Path,
    ) -> Result<(), CpuError> {
        Err(CpuError::NotSupported)
    }

    fn stop_capture(&self) -> Result<(), CpuError> {
        Err(CpuError::NotSupported)
    }

    fn device_capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::empty()
    }
}

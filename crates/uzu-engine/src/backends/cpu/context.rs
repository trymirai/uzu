use std::{
    path::Path,
    sync::{Arc, Weak, mpsc},
    thread,
};

use crate::backends::{
    common::{Allocation, AllocationPool, AllocationType, Allocator, Backend, Context, DeviceCapabilities},
    cpu::{Cpu, command_buffer::CpuCommandBufferEncoding, dense_buffer::CpuBuffer, error::CpuError},
};

pub struct CpuContext {
    pub(super) allocator: Arc<Allocator<Cpu>>,
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
            allocator: Allocator::new(weak_self.clone()),
            command_queue: command_queue_sender,
            weak_self: weak_self.clone(),
        }))
    }

    fn device_name(&self) -> Option<&str> {
        None
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<CpuBuffer, CpuError> {
        Ok(CpuBuffer::new(size))
    }

    fn create_allocation(
        &self,
        size: usize,
    ) -> Result<Allocation<Cpu>, CpuError> {
        self.allocator.allocate(size, AllocationType::Global)
    }

    fn create_allocation_pool(&self) -> Arc<AllocationPool<Cpu>> {
        Arc::new(self.allocator.create_pool())
    }

    fn create_command_buffer(
        &self,
        _name: Option<&str>,
        allocation_pool: Option<Arc<AllocationPool<Cpu>>>,
    ) -> Result<CpuCommandBufferEncoding, CpuError> {
        Ok(CpuCommandBufferEncoding::new(
            self.weak_self.upgrade().unwrap(),
            allocation_pool.unwrap_or_else(|| self.create_allocation_pool()),
        ))
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        Err(CpuError::NotSupported)
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

use std::{
    path::Path,
    sync::{Arc, mpsc},
    thread,
};

use crate::backends::{
    common::{Allocation, AllocationPool, AllocationType, Allocator, Backend, Context},
    cpu::{Cpu, command_buffer::CpuCommandBufferInitial, dense_buffer::CpuBuffer, error::CpuError},
};

pub struct CpuContext {
    allocator: Arc<Allocator<Cpu>>,
    command_queue: mpsc::Sender<Box<dyn FnOnce() + Send>>,
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
        }))
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
        allocation_type: AllocationType<Cpu>,
    ) -> Result<Allocation<Cpu>, CpuError> {
        self.allocator.allocate(size, allocation_type)
    }

    fn create_allocation_pool(
        &self,
        reusable: bool,
    ) -> AllocationPool<Cpu> {
        self.allocator.create_pool(reusable)
    }

    fn create_command_buffer(&self) -> Result<CpuCommandBufferInitial, CpuError> {
        Ok(CpuCommandBufferInitial::new(self.command_queue.clone()))
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

    fn sparse_buffers_supported(&self) -> bool {
        false
    }
}

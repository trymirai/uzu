use std::{
    cell::UnsafeCell,
    path::Path,
    pin::Pin,
    rc::Rc,
    sync::{atomic::AtomicU64, mpsc},
    thread,
};

use crate::backends::{
    common::{Allocation, AllocationPool, AllocationType, Allocator, Backend, Context},
    cpu::{Cpu, command_buffer::CpuCommandBufferInitial, error::CpuError},
};

pub struct CpuContext {
    allocator: Rc<Allocator<Cpu>>,
    command_queue: mpsc::Sender<Box<dyn FnOnce() + Send>>,
}

impl Context for CpuContext {
    type Backend = Cpu;

    fn new() -> Result<Rc<Self>, CpuError> {
        let (command_queue_sender, command_queue_receiever) = mpsc::channel::<Box<dyn FnOnce() + Send>>();

        thread::spawn(|| {
            for command_buffer in command_queue_receiever {
                command_buffer();
            }
        });

        Ok(Rc::new_cyclic(|weak_self| CpuContext {
            allocator: Allocator::new(weak_self.clone()),
            command_queue: command_queue_sender,
        }))
    }

    fn recommended_async_batch_size(
        &self,
        _model_path: &Path,
    ) -> usize {
        1
    }

    fn is_high_performance(&self) -> bool {
        false
    }

    fn debug_active(&self) -> bool {
        false
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<UnsafeCell<Pin<Box<[u8]>>>, CpuError> {
        Ok(UnsafeCell::new(Pin::new(vec![0; size].into_boxed_slice())))
    }

    fn create_sparse_buffer(
        &self,
        capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        todo!()
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

    fn create_event(&self) -> Result<Pin<Box<AtomicU64>>, CpuError> {
        Ok(Box::pin(AtomicU64::new(0)))
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }

    fn enable_capture() {}

    fn start_capture(
        &self,
        _trace_path: &std::path::Path,
    ) -> Result<(), CpuError> {
        Err(CpuError::NotSupported)
    }

    fn stop_capture(&self) -> Result<(), CpuError> {
        Err(CpuError::NotSupported)
    }
}

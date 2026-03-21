use std::{
    cell::{RefCell, UnsafeCell},
    path::Path,
    rc::Rc,
};

use super::{Cpu, command_buffer::CpuCommandBuffer, error::CpuError};
use crate::backends::common::{Allocation, AllocationPool, AllocationType, Allocator, Context};

pub struct CpuContext {
    allocator: Rc<Allocator<Cpu>>,
}

impl Context for CpuContext {
    type Backend = Cpu;

    fn new() -> Result<Rc<Self>, CpuError> {
        Ok(Rc::new_cyclic(|weak_self| CpuContext {
            allocator: Allocator::new(weak_self.clone()),
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
    ) -> Result<UnsafeCell<Box<[u8]>>, CpuError> {
        Ok(UnsafeCell::new(vec![0; size].into_boxed_slice()))
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

    fn create_command_buffer(&self) -> Result<CpuCommandBuffer, CpuError> {
        Ok(CpuCommandBuffer::new())
    }

    fn create_event(&self) -> Result<RefCell<u64>, CpuError> {
        Ok(RefCell::new(0))
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

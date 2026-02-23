use std::{cell::RefCell, path::Path, rc::Rc};

use crate::backends::{
    common::{Allocator, Backend, Context, DeviceClass},
    cpu::{backend::Cpu, buffer::CpuBuffer, command_buffer::CpuCommandBuffer, event::CpuEvent},
};

pub struct CpuContext {
    allocator: Allocator<Cpu>,
}

impl Context for CpuContext {
    type Backend = Cpu;

    fn new() -> Result<Rc<Self>, <Self::Backend as Backend>::Error> {
        Ok(Rc::new_cyclic(|weak_self| Self {
            allocator: Allocator::new(weak_self.clone()),
        }))
    }

    fn device_class(&self) -> DeviceClass {
        DeviceClass::Base
    }

    fn debug_active(&self) -> bool {
        false
    }

    fn allocator(&self) -> &Allocator<Self::Backend> {
        &self.allocator
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::NativeBuffer, <Self::Backend as Backend>::Error> {
        Ok(CpuBuffer::new(Box::from(vec![0; size])))
    }

    fn create_command_buffer(
        &self
    ) -> Result<<Self::Backend as Backend>::CommandBuffer, <Self::Backend as Backend>::Error> {
        Ok(CpuCommandBuffer::new())
    }

    fn create_event(&self) -> Result<<Self::Backend as Backend>::Event, <Self::Backend as Backend>::Error> {
        Ok(CpuEvent::new(RefCell::new(0)))
    }

    fn enable_capture() {}

    fn start_capture(
        &self,
        _trace_path: &Path,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        Ok(())
    }

    fn stop_capture(&self) -> Result<(), <Self::Backend as Backend>::Error> {
        Ok(())
    }
}

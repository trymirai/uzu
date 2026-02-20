use std::rc::Rc;

use crate::backends::{
    common::{Allocator, Backend, Context},
    cpu::backend::Cpu,
};

pub struct CpuContext {}

impl Context for CpuContext {
    type Backend = Cpu;

    fn new() -> Result<Rc<Self>, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn allocator(&self) -> &Allocator<Self::Backend> {
        todo!()
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::NativeBuffer, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn create_command_buffer(
        &self
    ) -> Result<<Self::Backend as Backend>::CommandBuffer, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn create_event(&self) -> Result<<Self::Backend as Backend>::Event, <Self::Backend as Backend>::Error> {
        todo!()
    }
}

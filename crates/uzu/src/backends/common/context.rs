use std::rc::Rc;

use super::{Allocator, Backend};

pub trait Context: Sized {
    type Backend: Backend<Context = Self>;

    fn new() -> Result<Rc<Self>, <Self::Backend as Backend>::Error>;

    fn allocator(&self) -> &Allocator<Self::Backend>;

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::NativeBuffer, <Self::Backend as Backend>::Error>;

    fn create_command_buffer(
        &self
    ) -> Result<<Self::Backend as Backend>::CommandBuffer, <Self::Backend as Backend>::Error>;

    fn create_event(&self) -> Result<<Self::Backend as Backend>::Event, <Self::Backend as Backend>::Error>;
}

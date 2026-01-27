use std::rc::Rc;

use super::Backend;

pub trait Context: Sized {
    type Backend: Backend;

    fn new() -> Result<Rc<Self>, <Self::Backend as Backend>::Error>;

    fn allocate_command_buffer(
        &self
    ) -> Result<
        <Self::Backend as Backend>::CommandBuffer,
        <Self::Backend as Backend>::Error,
    >;
}

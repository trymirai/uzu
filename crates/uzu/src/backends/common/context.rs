use std::{path::Path, rc::Rc};

use super::{Allocator, Backend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    Ultra,
    Max,
    Pro,
    Base,
    IPhone,
}

impl DeviceClass {
    pub fn is_high_end(&self) -> bool {
        matches!(self, DeviceClass::Ultra | DeviceClass::Max)
    }
}

pub trait Context: Sized {
    type Backend: Backend<Context = Self>;

    fn new() -> Result<Rc<Self>, <Self::Backend as Backend>::Error>;

    fn device_class(&self) -> DeviceClass;

    fn allocator(&self) -> &Allocator<Self::Backend>;

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::NativeBuffer, <Self::Backend as Backend>::Error>;

    fn create_command_buffer(
        &self
    ) -> Result<<Self::Backend as Backend>::CommandBuffer, <Self::Backend as Backend>::Error>;

    fn create_event(&self) -> Result<<Self::Backend as Backend>::Event, <Self::Backend as Backend>::Error>;

    fn enable_capture();

    fn start_capture(
        &self,
        trace_path: &Path,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn stop_capture(&self) -> Result<(), <Self::Backend as Backend>::Error>;
}

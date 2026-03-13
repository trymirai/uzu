use std::{env, path::Path, rc::Rc};

use super::Backend;
use crate::backends::common::CommandBuffer;

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

/// Device performance class based on the last character of architecture name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Phone,      // 'p' - iPhone/iPad integrated
    Integrated, // 'g' - Mac integrated GPU
    Desktop,    // 'd' - Mac Pro/Max discrete-class GPU
    Unknown(char),
}

impl DeviceType {
    pub fn is_high_performance(&self) -> bool {
        matches!(self, Self::Desktop)
    }
}

pub trait Context: Sized {
    type Backend: Backend<Context = Self>;

    fn new() -> Result<Rc<Self>, <Self::Backend as Backend>::Error>;

    fn device_class(&self) -> DeviceClass;

    fn device_type(&self) -> DeviceType;

    fn debug_active(&self) -> bool;

    fn create_command_buffer(
        &self
    ) -> Result<<<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Initial, <Self::Backend as Backend>::Error>;

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<<Self::Backend as Backend>::Buffer, <Self::Backend as Backend>::Error>;

    fn create_event(&self) -> Result<<Self::Backend as Backend>::Event, <Self::Backend as Backend>::Error>;

    fn enable_capture();

    fn start_capture(
        &self,
        trace_path: &Path,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn stop_capture(&self) -> Result<(), <Self::Backend as Backend>::Error>;

    fn tf32_enabled() -> bool {
        env::var("UZU_TF32").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
    }
}

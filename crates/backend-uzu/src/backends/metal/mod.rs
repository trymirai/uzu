mod backend;
mod buffer;
mod command_buffer;
mod context;
mod device_capabilities;
mod error;
mod event;
mod kernel;
mod metal_extensions;
mod sparse_pages;

pub use backend::Metal;
pub use context::MetalContext;
pub use device_capabilities::MetalDeviceCapabilities;
pub use kernel::matmul::MatmulDispatchPath;
pub use metal_extensions::{DeviceExt, DeviceGeneration};

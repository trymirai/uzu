mod backend;
mod buffer;
mod command_buffer;
mod context;
mod dense_buffer;
mod device_capabilities;
mod error;
mod kernel;
mod metal_extensions;
mod sparse;

pub use backend::Metal;
pub use buffer::BufferDowncastExt;
pub use context::MetalContext;
pub use device_capabilities::MetalDeviceCapabilities;
pub use kernel::matmul::{GemmDispatchPath, GemmKernel};
pub use metal_extensions::{DeviceExt, DeviceGeneration};

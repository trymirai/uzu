mod backend;
mod buffer;
mod command_buffer;
mod context;
mod device_capabilities;
mod error;
mod event;
mod kernel;
mod metal_extensions;

pub use backend::Metal;
pub use context::MetalContext;
pub use device_capabilities::MetalDeviceCapabilities;
pub use kernel::matmul::benchmark as matmul_benchmark;
pub use metal_extensions::DeviceGeneration;

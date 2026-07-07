mod backend;
mod buffer;
mod command_buffer;
mod context;
mod dense_buffer;
mod device_tier;
mod error;
mod kernel;
mod metal_extensions;
mod sparse;

pub use backend::Metal;
pub use context::MetalContext;
pub use kernel::matmul::GemmDispatchPath;
pub(crate) use metal_extensions::DeviceExt;

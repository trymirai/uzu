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
#[cfg(test)]
pub(crate) use device_tier::DeviceTier;
pub use kernel::matmul::GemmDispatchPath;
#[cfg(test)]
pub(crate) use kernel::matmul::gemv::{GemvDispatch, GemvSpecialization};

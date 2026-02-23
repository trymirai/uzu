mod backend;
mod buffer;
mod command_buffer;
mod context;
mod error;
mod event;
mod kernel;
mod metal_extensions;

pub use backend::Metal;
pub use context::MetalContext;
pub use kernel::matmul::choose_dispatch_descriptor;

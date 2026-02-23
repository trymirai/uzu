mod backend;
mod command_buffer;
mod context;
mod copy_encoder;
mod error;
mod event;
mod kernel;
mod metal_extensions;
mod native_buffer;

pub use backend::Metal;
pub use kernel::matmul::choose_dispatch_descriptor;

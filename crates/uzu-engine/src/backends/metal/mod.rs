mod backend;
mod buffer;
mod command_buffer;
mod context;
mod dense_buffer;
mod error;
mod kernel;
mod metal_extensions;
mod sparse;

pub use backend::Metal;
// TODO: This should be removed
pub use context::MetalContext;
#[cfg(test)]
pub use kernel::matmul::gemm::GemmEngine;

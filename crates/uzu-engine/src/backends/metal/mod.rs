mod backend;
mod buffer;
mod command_buffer;
mod context;
mod decompression;
mod error;
mod kernel;
mod metal_extensions;

pub use backend::Metal;
pub use context::MetalContext; // TODO: This should be removed
#[cfg(test)]
pub use kernel::matmul::gemm::GemmEngine;

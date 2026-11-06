mod backend;
mod buffer;
mod command_buffer;
mod context;
mod error;
mod kernel;

pub use backend::WebGPU;

#[cfg(test)]
#[path = "../../../unit/backends/webgpu/context_test.rs"]
mod tests;

mod backend;
mod command_buffer;
mod context;
mod dense_buffer;
mod error;
mod event;
mod kernel;

pub use backend::Cpu;

pub mod argmax;
/// Re-export of the NF4-E4M3 CPU reference helpers for the bench/correctness
/// test crate (the NF4 GPU kernels are bench-only with no PUBLIC CPU pair).
pub use kernel::nf4_e4m3;

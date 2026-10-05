//! AMD GPUs through the HIP runtime. Kernels are uzu's MSL sources compiled by clang for AMDGPU
//! (see `build/amdgpu`); memory is host-mapped, which matches the unified-memory contract of
//! `GlobalBuffer: BufferCpuAccessible` on APUs.

mod backend;
pub(crate) mod buffer;
mod command_buffer;
pub(crate) mod context;
pub(crate) mod error;
mod hip;
pub(crate) mod kernel;
mod profile;

pub use backend::Amdgpu;

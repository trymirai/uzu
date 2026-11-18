pub(crate) mod common;

#[cfg(backend = "metal")]
pub(crate) mod metal;

#[cfg(backend = "vulkan")]
pub mod vulkan;

#[cfg(backend = "webgpu")]
pub(crate) mod webgpu;

#[cfg(backend = "cpu")]
pub(crate) mod cpu;

mod backend_selection;

pub use backend_selection::{BackendSelection, select_backend, selected_backend};
pub use common::Backend;

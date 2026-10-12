mod backend_selection;

pub mod common;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(backend = "cpu")]
pub mod cpu;
#[cfg(backend = "metal")]
pub mod metal;

pub use backend_selection::{BackendSelection, select_backend};

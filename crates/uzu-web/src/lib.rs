//! Browser bindings run in a dedicated worker so the synchronous engine can use
//! the WebGPU backend without blocking the page's event loop.

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod chat;
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod encoding;

#[cfg(target_arch = "wasm32")]
mod encoded_run;
#[cfg(target_arch = "wasm32")]
mod session;

#[cfg(target_arch = "wasm32")]
pub use session::{Session, create_session};

#[cfg(all(target_arch = "wasm32", feature = "kernel-tests"))]
mod gdn_tests;
#[cfg(all(target_arch = "wasm32", feature = "kernel-tests"))]
mod gemv_tests;
#[cfg(all(target_arch = "wasm32", feature = "kernel-tests"))]
mod kernel_tests;
#[cfg(all(target_arch = "wasm32", feature = "kernel-tests"))]
mod sampling_tests;
#[cfg(all(target_arch = "wasm32", feature = "kernel-tests"))]
mod shader_cache_tests;
#[cfg(all(target_arch = "wasm32", feature = "kernel-tests"))]
mod weight_source_tests;

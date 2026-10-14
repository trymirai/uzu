//! WebGPU execution from a browser worker.
//!
//! The inference engine remains synchronous. GPU work runs in a second worker;
//! the engine worker waits through shared completion signals, leaving the GPU
//! worker's event loop free to resolve WebGPU promises.

mod backend;
pub mod buffer;
pub mod command_buffer;
pub mod context;
mod error;
pub(crate) mod kernel;
mod model_config;
mod runtime;
mod weight_source;

pub use backend::WebGpu;
pub use error::WebGpuError;
pub use model_config::validate_model_config;
pub use weight_source::WebGpuWeightSource;

/// Starts the GPU worker without blocking its creator's event loop. Call once
/// before each synchronous engine/context construction in a browser worker.
pub async fn prepare() -> Result<(), WebGpuError> {
    runtime::prepare_context().await.map_err(Into::into)
}

/// Diagnostic counters, enabled by setting `globalThis.UZU_WEBGPU_PROFILE`
/// before preparing the context. GPU timestamps are requested only in this mode.
pub fn profile_snapshot(reset: bool) -> String {
    runtime::profile_snapshot(reset)
}

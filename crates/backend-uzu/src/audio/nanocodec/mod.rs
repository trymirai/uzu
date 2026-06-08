#[cfg(metal_backend)]
pub mod runtime;

#[cfg(all(metal_backend, test))]
pub use runtime::NanoCodecFsqRuntimeConfig;
#[cfg(metal_backend)]
pub use runtime::{AudioDecodeStepStats, AudioDecodeStreamState, NanoCodecFsqRuntime};

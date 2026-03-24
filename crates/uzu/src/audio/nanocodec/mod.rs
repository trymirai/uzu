#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
pub mod runtime;

#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
pub use runtime::{AudioDecodeStepStats, AudioDecodeStreamState, NanoCodecFsqRuntime, NanoCodecFsqRuntimeConfig};

#[cfg(metal_backend)]
pub mod runtime;

#[cfg(metal_backend)]
pub use runtime::{AudioDecodeStepStats, AudioDecodeStreamState, NanoCodecFsqRuntime, NanoCodecFsqRuntimeConfig};

#[cfg(test)]
#[path = "../../../tests/unit/audio/nanocodec/mod.rs"]
mod tests;

mod codec;
#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
mod context;
pub mod nanocodec;
mod types;

pub use codec::AudioCodecRuntime;
#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
pub use context::AudioGenerationContext;
#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
pub use nanocodec::{NanoCodecFsqRuntime, NanoCodecFsqRuntimeConfig};
pub use types::{AudioError, AudioPcmBatch, AudioResult, AudioTokenGrid};

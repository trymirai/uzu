mod codec;
#[cfg(feature = "audio-runtime")]
mod context;
pub mod nanocodec;
mod types;

pub use codec::AudioCodecRuntime;
#[cfg(feature = "audio-runtime")]
pub use context::AudioGenerationContext;
#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
pub use nanocodec::{
    NanoCodecFsqRuntime, NanoCodecFsqRuntimeConfig, NanoCodecFsqRuntimeOptions,
};
pub use types::{AudioError, AudioPcmBatch, AudioResult, AudioTokenGrid, AudioTokenPacking};

#[cfg(metal_backend)]
mod codec;
#[cfg(metal_backend)]
mod context;
pub mod nanocodec;
mod types;

#[cfg(metal_backend)]
pub use codec::AudioCodecRuntime;
#[cfg(metal_backend)]
pub use context::AudioGenerationContext;
#[cfg(metal_backend)]
pub use nanocodec::NanoCodecFsqRuntime;
#[cfg(all(metal_backend, test))]
pub use nanocodec::NanoCodecFsqRuntimeConfig;
pub use types::AudioError;
#[cfg(metal_backend)]
pub use types::{AudioPcmBatch, AudioResult, AudioTokenGrid};

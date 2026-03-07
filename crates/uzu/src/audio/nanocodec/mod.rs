pub mod decoder;
pub mod fsq;
pub mod ops;

#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
pub mod runtime;

#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
pub use runtime::{
    AudioCommandBufferProfile, AudioDecodeProfile, AudioDecodeStepStats, AudioDecodeStreamState,
    AudioDecodeStreamingMode, NanoCodecFsqRuntime, NanoCodecFsqRuntimeConfig, NanoCodecFsqRuntimeOptions,
};

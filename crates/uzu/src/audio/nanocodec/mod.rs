pub mod fsq;

#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
pub mod runtime;

#[cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]
pub use runtime::{
    AudioDecodeStepStats, AudioDecodeStreamState, AudioDecodeStreamingMode, NanoCodecFsqRuntime,
    NanoCodecFsqRuntimeConfig, NanoCodecFsqRuntimeOptions,
};

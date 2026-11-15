#[cfg(backend = "metal")]
pub mod config;
#[cfg(backend = "metal")]
pub mod dry;
#[cfg(backend = "metal")]
pub mod linear;
#[cfg(backend = "metal")]
pub mod meanflow_head;
#[cfg(backend = "metal")]
pub mod session;
#[cfg(backend = "metal")]
pub mod vibevoice_codec;

#[cfg(backend = "metal")]
pub use config::{HaruCodecConfig, HaruConfig, HaruDryDefaults, HaruGenerationDefaults, HaruHeadConfig};
#[cfg(backend = "metal")]
pub use dry::DryState;
#[cfg(backend = "metal")]
pub use meanflow_head::MeanFlowHead;
#[cfg(backend = "metal")]
pub use session::{HaruDuplexSession, HaruFrame, HaruPipeline, HaruSessionConfig};
#[cfg(backend = "metal")]
pub use vibevoice_codec::{CodecDecoderState, CodecEncoderState, VibeVoiceCodec};

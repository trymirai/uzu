#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod lfm3_omni;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use lfm3_omni::{Lfm3OmniPipeline, OmniMessage, OmniVideoInput, group_seconds, prepare_conversation_parts, wrap_media};

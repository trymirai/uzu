#[cfg(feature = "bindings-uniffi")]
uniffi::setup_scaffolding!();

pub mod api;
pub mod chat;
pub mod classification;
pub mod text_to_speech;

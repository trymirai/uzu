#[cfg(feature = "bindings-uniffi")]
uniffi::setup_scaffolding!();

pub mod api;
pub mod chat;
pub mod classification;
pub mod telemetry;
pub mod text_to_speech;

mod tool;
mod util;

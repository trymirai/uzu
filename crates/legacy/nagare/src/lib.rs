#[cfg(feature = "bindings-uniffi")]
uniffi::setup_scaffolding!();

#[cfg(not(target_family = "wasm"))]
pub mod api;
pub mod chat;
pub mod classification;
pub mod telemetry;
pub mod text_to_speech;
pub mod tool;
mod util;

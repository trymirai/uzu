#[cfg(feature = "bindings-uniffi")]
uniffi::setup_scaffolding!();

pub mod api;
pub mod chat;
#[cfg(not(target_family = "wasm"))]
pub mod classification;
#[cfg(not(target_family = "wasm"))]
pub mod text_to_speech;
pub mod tool;
mod util;

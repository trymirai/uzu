#[cfg(feature = "bindings-uniffi")]
shoji::uniffi_reexport_scaffolding!();
#[cfg(feature = "bindings-uniffi")]
nagare::uniffi_reexport_scaffolding!();
#[cfg(feature = "bindings-uniffi")]
uniffi::setup_scaffolding!();

#[cfg(not(target_family = "wasm"))]
pub mod device;
#[cfg(not(target_family = "wasm"))]
pub mod engine;
#[cfg(not(target_family = "wasm"))]
pub mod helpers;
#[cfg(not(target_family = "wasm"))]
pub mod logs;
#[cfg(not(target_family = "wasm"))]
pub mod registry;
#[cfg(not(target_family = "wasm"))]
pub mod storage;

pub use shoji::*;

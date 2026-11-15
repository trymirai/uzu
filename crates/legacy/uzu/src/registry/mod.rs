mod cached;
mod error;
#[cfg(not(target_family = "wasm"))]
pub mod hf_cache;
pub mod local;
mod merged;
pub mod mirai;
#[cfg(not(target_family = "wasm"))]
pub mod openai;

pub use cached::CachedRegistry;
pub use error::RegistryError;
pub use merged::MergedRegistry;
pub(crate) use merged::unique_model;

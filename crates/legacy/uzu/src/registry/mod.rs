mod cached;
mod error;
pub mod local;
mod merged;
pub mod mirai;
#[cfg(not(target_family = "wasm"))]
pub mod openai;

pub use cached::CachedRegistry;
pub use error::RegistryError;
pub use merged::MergedRegistry;
pub(crate) use merged::unique_model;

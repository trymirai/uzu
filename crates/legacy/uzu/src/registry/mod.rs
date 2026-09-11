mod cached;
mod error;
pub mod local;
mod merged;
pub mod mirai;
pub mod openai;

pub use cached::CachedRegistry;
pub use error::RegistryError;
pub use merged::MergedRegistry;

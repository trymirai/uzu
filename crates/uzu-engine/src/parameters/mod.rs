// TODO: This is overdue for a complete rewrite

mod error;
mod loader;
mod safetensors_metadata;

pub use error::ParameterLoaderError;
pub use loader::{ParameterLoader, ParameterTree};

// TODO: This is overdue for a complete rewrite

mod error;
mod loader;
#[cfg(backend = "metal")]
mod mxu_cache;
mod safetensors_metadata;

const MXU_LAYOUT: &str = "w4_n32_k128_v1";

pub use error::ParameterLoaderError;
pub use loader::{ParameterLoader, ParameterTree};

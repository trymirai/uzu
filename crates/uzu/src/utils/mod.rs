pub mod env_utils;
#[cfg(metal_backend)]
pub mod model_size;
pub mod pointers;
pub mod version;

pub use version::VERSION;

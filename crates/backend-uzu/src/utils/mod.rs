pub mod env_utils;
pub mod fs;
pub mod memory;
#[cfg(metal_backend)]
pub mod model_size;
pub mod pointers;
mod sparse_array;
pub mod version;

pub use sparse_array::{SparseArray, SparseArrayContext};
pub use version::VERSION;

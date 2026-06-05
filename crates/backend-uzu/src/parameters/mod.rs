mod loader;
mod safetensors_metadata;

// Re-export the safetensors header reader so other modules (e.g. decoder
// runner) can estimate parameter memory before creating a Context.
pub use loader::{ParameterLeaf, ParameterLoader, ParameterLoaderError, ParameterTree};
pub use safetensors_metadata::HeaderLoadingError;
#[cfg(test)]
pub use safetensors_metadata::read_metadata;

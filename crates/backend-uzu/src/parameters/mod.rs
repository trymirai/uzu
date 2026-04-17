mod loader;
mod safetensors_metadata;

// Re-export the safetensors header reader so other modules (e.g. decoder
// runner) can estimate parameter memory before creating a Context.
pub use loader::{ParameterLeaf, ParameterLoader, ParameterLoaderError, ParameterTree, resolve_subtree};
pub use safetensors_metadata::{HeaderLoadingError, read_metadata as read_safetensors_metadata};

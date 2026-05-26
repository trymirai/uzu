mod loader;
mod safetensors_metadata;

// Re-export the safetensors header reader so other modules (e.g. decoder
// runner) can estimate parameter memory before creating a Context.
pub use loader::{ParameterLeaf, ParameterLoader, ParameterLoaderError, ParameterTree};
pub use safetensors_metadata::{
    Dtype, HashMetadata, HeaderLoadingError, SafeTensorData, read_metadata as read_safetensors_metadata,
    write_safetensors, write_safetensors_with_metadata,
};

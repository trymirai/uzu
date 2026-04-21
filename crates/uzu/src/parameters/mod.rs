mod loader;
pub use loader::{ParameterLeaf, ParameterLoader, ParameterLoaderError, ParameterTree, resolve_subtree};
mod safetensors_metadata;

// Re-export the safetensors header reader so other modules (e.g. decoder
// runner) can estimate parameter memory before creating a Context.
pub use safetensors_metadata::{
    Dtype as SafetensorsDtype, HashMetadata as SafetensorsMetadata, HeaderLoadingError, SafetensorsWriteError,
    TensorData, TensorInfo as SafetensorsTensorInfo, read_metadata as read_safetensors_metadata, write_safetensors,
};

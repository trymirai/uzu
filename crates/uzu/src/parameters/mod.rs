mod loader;
pub use loader::{ParameterLoader, ParameterLoaderError, ParameterTree};
mod safetensors_metadata;
mod safetensors_writer;

// Re-export the safetensors header reader so other modules (e.g. decoder
// runner) can estimate parameter memory before creating a DeviceContext.
pub use safetensors_metadata::read_metadata as read_safetensors_metadata;
pub use safetensors_metadata::{Dtype, HashMetadata, TensorInfo};
pub use safetensors_writer::{
    SafetensorHeaderEntry, SafetensorView, SafetensorsWriteError,
    write_safetensors, write_safetensors_streaming,
};

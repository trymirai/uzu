mod loader;
pub use loader::{ParameterLoader, ParameterLoaderError, ParameterTree};
mod safetensors_metadata;

// Re-export the safetensors header reader so other modules (e.g. decoder
// runner) can estimate parameter memory before creating a DeviceContext.
pub use safetensors_metadata::read_metadata as read_safetensors_metadata;

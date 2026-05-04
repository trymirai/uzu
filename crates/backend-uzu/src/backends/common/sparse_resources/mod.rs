mod sparse_buffer;
mod sparse_buffer_operation;
mod sparse_resource_mapping_mode;

#[cfg(metal_backend)]
pub(crate) use sparse_buffer::SparseBufferMappedPages;
pub use sparse_buffer::{SparseBuffer, SparseBufferExt};
pub use sparse_buffer_operation::SparseBufferOperation;
pub use sparse_resource_mapping_mode::SparseResourceMappingMode;

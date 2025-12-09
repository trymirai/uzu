mod array;
mod buffer_allocator;
mod classifier_context;
pub mod compilation_parameters;
mod context;
pub mod encodable_block;
pub mod error;
pub mod forward_pass;
pub mod graph;
pub mod image;
pub mod kernel;
mod llm_context;
pub mod media_utils;
pub mod metal_extensions;
pub mod placement_analysis;
pub mod utils;

pub use array::MetalArray;
pub use buffer_allocator::BufferAllocator;
pub use classifier_context::ClassifierContext;
pub use context::MTLContext;
pub use encodable_block::Decoder;
pub use error::MTLError;
pub use forward_pass::{CacheLayers, ForwardPassState, ModelShape};
pub use kernel::{KVCacheUpdate, KernelDataType, RopeKernel};
pub use llm_context::LLMContext;
pub use media_utils::{
    ImagePreprocessingParams, ImagePreprocessingRequirements,
    MetalImagePreprocessor,
};
pub use placement_analysis::PlacementAnalysis;

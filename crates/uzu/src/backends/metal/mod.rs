mod array;
mod buffer_allocator;
pub mod compilation_parameters;
mod context;
pub mod error;
mod executable_builder;
pub mod forward_pass;
pub mod graph;
pub mod image;
pub mod kernel;
pub mod media_utils;
pub mod metal_extensions;
pub mod placement_analysis;
pub mod utils;
pub use array::MetalArray;
pub use buffer_allocator::BufferAllocator;
pub use context::MTLContext;
pub use error::MTLError;
pub use executable_builder::DecoderExecutables;
pub use forward_pass::{CacheLayers, ForwardPassState, ModelShape};
pub use kernel::{AsyncScatterKV, KVCacheUpdate, KernelDataType, RopeKernel};
// pub use kernel::{
//     DataType as KernelDataType, Encoder as KernelEncoder,
//     EncoderError as KernelEncoderError, TensorAddSwap, TensorCopy,
// };
pub use media_utils::{
    ImagePreprocessingParams, ImagePreprocessingRequirements,
    MetalImagePreprocessor,
};
pub use placement_analysis::PlacementAnalysis;

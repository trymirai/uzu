mod array;
mod backend;
mod classifier_context;
mod command_buffer;
pub mod compilation_parameters;
mod context;
pub mod encodable_block;
pub mod error;
pub mod forward_pass;
pub mod image;
pub mod kernel;
mod language_model_generator_context;
pub mod media_utils;
pub mod metal_extensions;
pub mod placement_analysis;
pub mod utils;

pub use metal::prelude::*;

pub use array::MetalArray;
pub use backend::Metal;
pub use classifier_context::ClassifierContext;
pub use context::{
    DeviceArchitecture, DeviceClass, DeviceGeneration, MTLContext,
};
pub use encodable_block::Decoder;
pub use error::MTLError;
pub use forward_pass::{CacheLayers, ForwardPassState, ModelShape};
pub use kernel::{KVCacheUpdate, KernelDataType, MetalKernels, RopeKernel};
pub use language_model_generator_context::LanguageModelGeneratorContext;
pub use media_utils::{
    ImagePreprocessingParams, ImagePreprocessingRequirements,
    MetalImagePreprocessor,
};
pub use metal_extensions::ComputeEncoderSetValue;
pub use placement_analysis::PlacementAnalysis;

mod array;
mod buffer_allocator;
mod classifier_context;
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

// Re-export mtl-rs types with convenient type aliases
pub use metal::{
    MTLBindingExt, MTLBlitCommandEncoder, MTLBuffer, MTLCaptureDescriptor,
    MTLCaptureDestination, MTLCaptureManager, MTLCommandBuffer,
    MTLCommandBufferExt, MTLCommandEncoder, MTLCommandEncoderExt,
    MTLCommandQueue, MTLCommandQueueExt, MTLCompareFunction,
    MTLComputeCommandEncoder, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLDeviceExt, MTLEvent,
    MTLFeatureSet, MTLFunctionConstantValues, MTLGPUFamily, MTLHeap,
    MTLHeapExt, MTLLibrary, MTLLibraryExt, MTLPipelineOption, MTLPixelFormat,
    MTLReadWriteTextureTier, MTLResource, MTLResourceExt, MTLResourceOptions,
    MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};
pub use objc2::{rc::Retained, runtime::ProtocolObject};
pub use objc2_foundation::NSRange;

/// Type alias for owned MTLComputeCommandEncoder
pub type ComputeCommandEncoder =
    Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// Type alias for owned MTLComputePipelineState
pub type ComputePipelineState =
    Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Type alias for owned MTLCommandQueue
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

/// Type alias for owned MTLDevice
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Type alias for owned MTLHeap
pub type Heap = Retained<ProtocolObject<dyn MTLHeap>>;

/// Type alias for owned MTLLibrary  
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// Type alias for owned MTLEvent
pub type Event = Retained<ProtocolObject<dyn MTLEvent>>;

/// Type alias for owned MTLTexture
pub type Texture = Retained<ProtocolObject<dyn MTLTexture>>;

/// Type alias for MTLFunctionConstantValues
pub type FunctionConstantValues = MTLFunctionConstantValues;

/// Type alias for MTLTextureDescriptor  
pub type TextureDescriptor = MTLTextureDescriptor;

pub use array::MetalArray;
pub use buffer_allocator::BufferAllocator;
pub use classifier_context::ClassifierContext;
pub use context::{
    DeviceArchitecture, DeviceClass, DeviceGeneration, MTLContext,
};
pub use encodable_block::Decoder;
pub use error::MTLError;
pub use forward_pass::{CacheLayers, ForwardPassState, ModelShape};
pub use kernel::{KVCacheUpdate, KernelDataType, RopeKernel};
pub use language_model_generator_context::LanguageModelGeneratorContext;
pub use media_utils::{
    ImagePreprocessingParams, ImagePreprocessingRequirements,
    MetalImagePreprocessor,
};
pub use metal_extensions::{BufferLabelExt, FunctionConstantValuesLegacy};
pub use placement_analysis::PlacementAnalysis;

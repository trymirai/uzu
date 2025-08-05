use thiserror::Error;

#[derive(Debug, Error)]
pub enum MTLError {
    #[error("MTLContext error: {0}")]
    Context(#[from] ContextError),
    #[error("MTLDevice error: {0}")]
    Device(#[from] DeviceError),
    #[error("MTLHeap error: {0}")]
    Heap(#[from] HeapError),
    #[error("MTLCommandBuffer error: {0}")]
    CommandBuffer(#[from] CommandBufferError),
    #[error("MTLCommandQueue error: {0}")]
    CommandQueue(#[from] CommandQueueError),
    #[error("MTLLibrary error: {0}")]
    Library(#[from] LibraryError),
    #[error("MTLTexture error: {0}")]
    Texture(#[from] TextureError),
    #[error("MTLTexture serialization error: {0}")]
    TextureSerialization(#[from] TextureSerializationError),
    #[error("MTLResource error: {0}")]
    Resource(#[from] ResourceError),
    #[error("MTLBuffer error: {0}")]
    Buffer(#[from] BufferError),
    #[error("MTLPixelFormat error: {0}")]
    PixelFormat(#[from] PixelFormatError),
    #[error("Generic Metal error: {0}")]
    Generic(String),
}

impl From<&str> for MTLError {
    fn from(error: &str) -> Self {
        MTLError::Generic(error.to_string())
    }
}

impl From<String> for MTLError {
    fn from(error: String) -> Self {
        MTLError::Generic(error)
    }
}

// Context errors
#[derive(Debug, Error)]
pub enum ContextError {
    #[error("Failed to create texture cache")]
    TextureCacheCreationFailed,
    #[error("{0}")]
    Custom(String),
}

// Device errors
#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("Failed to create argument encoder")]
    ArgumentEncoderCreationFailed,
    #[error("Failed to create buffer")]
    BufferCreationFailed,
    #[error("Failed to create command queue")]
    CommandQueueCreationFailed,
    #[error("Failed to create depth stencil state")]
    DepthStencilStateCreationFailed,
    #[error("Failed to create event")]
    EventCreationFailed,
    #[error("Failed to create fence")]
    FenceCreationFailed,
    #[error("Failed to create heap")]
    HeapCreationFailed,
    #[error("Failed to create indirect command buffer")]
    IndirectCommandBufferCreationFailed,
    #[error("Failed to create library")]
    LibraryCreationFailed,
    #[error("Failed to create rasterization rate map")]
    RasterizationRateMapCreationFailed,
    #[error("Failed to create sampler state")]
    SamplerStateCreationFailed,
    #[error("Failed to create texture")]
    TextureCreationFailed,
    #[error("Failed to create texture view")]
    TextureViewCreationFailed,
    #[error("{0}")]
    Custom(String),
}

// Heap errors
#[derive(Debug, Error)]
pub enum HeapError {
    #[error("Failed to create buffer from heap")]
    BufferCreationFailed,
    #[error("Failed to create texture from heap")]
    TextureCreationFailed,
    #[error("{0}")]
    Custom(String),
}

// Command buffer errors
#[derive(Debug, Error)]
pub enum CommandBufferError {
    #[error("Failed to execute command buffer")]
    ExecutionFailed,
    #[error("{0}")]
    Custom(String),
}

// Command queue errors
#[derive(Debug, Error)]
pub enum CommandQueueError {
    #[error("Failed to create command buffer")]
    CommandBufferCreationFailed,
    #[error("{0}")]
    Custom(String),
}

// Library errors
#[derive(Debug, Error)]
pub enum LibraryError {
    #[error("Failed to create function")]
    FunctionCreationFailed,
    #[error("Failed to create pipeline state")]
    PipelineStateCreationFailed,
    #[error("{0}")]
    Custom(String),
}

// Texture errors
#[derive(Debug, Error)]
pub enum TextureError {
    #[error("Failed to create image from texture")]
    ImageCreationFailed,
    #[error("Incompatible pixel format")]
    IncompatiblePixelFormat,
    #[error("{0}")]
    Custom(String),
}

// Texture serialization errors
#[derive(Debug, Error)]
pub enum TextureSerializationError {
    #[error("Failed to allocate memory for texture serialization")]
    AllocationFailed,
    #[error("Failed to access texture data")]
    DataAccessFailure,
    #[error("Unsupported pixel format for serialization")]
    UnsupportedPixelFormat,
    #[error("{0}")]
    Custom(String),
}

// Resource errors
#[derive(Debug, Error)]
pub enum ResourceError {
    #[error("Resource unavailable")]
    ResourceUnavailable,
    #[error("{0}")]
    Custom(String),
}

// Buffer errors
#[derive(Debug, Error)]
pub enum BufferError {
    #[error("Incompatible data for buffer")]
    IncompatibleData,
    #[error("Failed to create texture from buffer")]
    TextureCreationFailed,
    #[error("{0}")]
    Custom(String),
}

// Pixel format errors
#[derive(Debug, Error)]
pub enum PixelFormatError {
    #[error("Incompatible CV pixel format")]
    IncompatibleCVPixelFormat,
    #[error("{0}")]
    Custom(String),
}

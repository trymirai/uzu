use thiserror::Error;

#[derive(Debug, Error)]
pub enum WebGPUError {
    #[error("Not supported")]
    NotSupported,
    #[error("Request adapter error: {0}")]
    RequestAdapter(#[from] wgpu::RequestAdapterError),
    #[error("Request device error: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("Map range error: {0}")]
    MapRange(#[from] wgpu::MapRangeError),
    #[error("Async map error: {0}")]
    AsyncMap(#[from] wgpu::BufferAsyncError),
    #[error("Command buffer failed")]
    CommandBuffer,
    #[error("Unsupported kernel variant: {0}")]
    UnsupportedKernelVariant(String),
}

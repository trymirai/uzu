use thiserror::Error;

#[derive(Debug, Error)]
pub enum WebGPUError {
    #[error("Not supported")]
    NotSupported,
    #[error("Request adapter error: {0}")]
    RequestAdapterError(#[from] wgpu::RequestAdapterError),
    #[error("Request device error: {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("Map range error: {0}")]
    MapRangeError(#[from] wgpu::MapRangeError),
    #[error("Command buffer failed")]
    CommandBufferFailed,
    #[error("UnsupportedKernelVariant: {0}")]
    UnsupportedKernelVariant(String),
}

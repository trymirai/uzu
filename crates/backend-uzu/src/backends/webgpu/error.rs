use thiserror::Error;

#[derive(Debug, Error)]
pub enum WebGPUError {
    #[error("Request adapter error: {0}")]
    RequestAdapterError(#[from] wgpu::RequestAdapterError),
    #[error("Request device error: {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("Command buffer failed")]
    CommandBufferFailed,
}

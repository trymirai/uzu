use std::sync::Arc;

use parking_lot::Mutex;
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
    #[error("WebGPU device error: {0}")]
    Device(String),
}

/// WebGPU reports validation, out-of-memory and device-lost errors asynchronously, not from the call that caused them.
/// This keeps the earliest unreported one until a GPU wait returns it.
#[derive(Clone, Default)]
pub(super) struct DeviceErrors(Arc<Mutex<Option<String>>>);

impl DeviceErrors {
    pub(super) fn record(
        &self,
        message: String,
    ) {
        self.0.lock().get_or_insert(message);
    }

    pub(super) fn take(&self) -> Result<(), WebGPUError> {
        match self.0.lock().take() {
            Some(message) => Err(WebGPUError::Device(message)),
            None => Ok(()),
        }
    }
}

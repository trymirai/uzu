use std::io;

use wasm_bindgen::JsValue;

use super::{WebGpuError, buffer::WebGpuBuffer, context::WebGpuContext, runtime};
use crate::parameters::ParameterReadSource;

/// A Blob or Uint8Array borrowed by a synchronous browser model load. Neither
/// the checkpoint nor GPU-only parameter tensors are copied into WASM memory.
pub struct WebGpuWeightSource {
    value: JsValue,
    length: u64,
    cancellation: JsValue,
}

impl WebGpuWeightSource {
    pub fn new(value: JsValue) -> Result<Self, WebGpuError> {
        let length = runtime::weight_source_size(&value)?;
        if !length.is_finite() || length.fract() != 0.0 || !(1.0..=9_007_199_254_740_991.0).contains(&length) {
            return Err(WebGpuError::Runtime("Invalid weights source length".into()));
        }
        Ok(Self {
            value,
            length: length as u64,
            cancellation: JsValue::UNDEFINED,
        })
    }

    /// A shared flag permits the caller to cancel while this worker is blocked
    /// in synchronous loading. No flag is required for existing callers.
    pub fn with_cancellation(
        mut self,
        cancellation: JsValue,
    ) -> Result<Self, WebGpuError> {
        runtime::check_load_cancellation(&cancellation)?;
        self.cancellation = cancellation;
        Ok(self)
    }

    pub fn check_cancellation(&self) -> Result<(), WebGpuError> {
        runtime::check_load_cancellation(&self.cancellation).map_err(Into::into)
    }

    fn validate_range(
        &self,
        offset: u64,
        size: usize,
    ) -> io::Result<()> {
        if offset.checked_add(size as u64).is_none_or(|end| end > self.length) {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Weights read extends beyond the source"));
        }
        Ok(())
    }

    pub fn initialize_buffer(
        &self,
        context: &WebGpuContext,
        offset: u64,
        size: usize,
    ) -> Result<WebGpuBuffer, WebGpuError> {
        self.check_cancellation()?;
        self.validate_range(offset, size).map_err(|error| WebGpuError::Runtime(error.to_string()))?;
        WebGpuBuffer::from_source(context.runtime.clone(), &self.value, offset, size, &self.cancellation)
    }
}

impl ParameterReadSource for WebGpuWeightSource {
    fn len(&self) -> u64 {
        self.length
    }

    fn read_exact_at(
        &self,
        destination: &mut [u8],
        offset: u64,
    ) -> io::Result<()> {
        self.validate_range(offset, destination.len())?;
        runtime::read_weight_source(&self.value, offset as f64, destination, &self.cancellation)
            .map_err(|error| io::Error::other(WebGpuError::from(error).to_string()))
    }
}

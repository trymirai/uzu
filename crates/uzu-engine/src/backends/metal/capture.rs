use std::path::Path;

use metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLCaptureTarget};

use crate::backends::metal::error::MetalError;

pub struct MetalCaptureSpan;

impl MetalCaptureSpan {
    pub fn new(
        target: &MTLCaptureTarget,
        output_path: &Path,
    ) -> Result<Self, MetalError> {
        let capture_descriptor = MTLCaptureDescriptor::new();
        capture_descriptor.set_capture_object(Some(target));
        capture_descriptor.set_destination(MTLCaptureDestination::GPUTraceDocument);
        capture_descriptor.set_output_path(Some(output_path));

        MTLCaptureManager::shared_capture_manager()
            .start_capture_with_descriptor(&capture_descriptor)
            .map_err(|error| MetalError::CannotStartGpuCapture(error.to_string()))?;

        Ok(Self)
    }
}

impl Drop for MetalCaptureSpan {
    fn drop(&mut self) {
        MTLCaptureManager::shared_capture_manager().stop_capture();
    }
}

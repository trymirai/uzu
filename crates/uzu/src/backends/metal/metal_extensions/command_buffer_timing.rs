use metal::{MTLCommandBuffer, MTLCommandBufferExt};
use objc2::runtime::ProtocolObject;

/// Extension trait providing convenience timing methods for command buffers.
pub trait CommandBufferTimingExt {
    /// Returns the GPU execution time in milliseconds.
    ///
    /// This is a convenience method that calculates `(kernel_end_time - kernel_start_time) * 1000`.
    fn gpu_execution_time_ms(&self) -> Option<f64>;
}

impl CommandBufferTimingExt for ProtocolObject<dyn MTLCommandBuffer> {
    fn gpu_execution_time_ms(&self) -> Option<f64> {
        match (self.kernel_start_time(), self.kernel_end_time()) {
            (Some(start), Some(end)) => Some((end - start) * 1000.0),
            _ => None,
        }
    }
}

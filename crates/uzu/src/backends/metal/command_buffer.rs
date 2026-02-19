use std::ops::Deref;

use metal::{MTLCommandBuffer, MTLCommandBufferExt, MTLCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::backends::common::CommandBuffer;

impl CommandBuffer for Retained<ProtocolObject<dyn MTLCommandBuffer>> {
    type Backend = Metal;

    fn with_compute_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as crate::backends::common::Backend>::ComputeEncoder) -> T,
    ) -> T {
        let encoder = self.new_compute_command_encoder().expect("Failed to create compute command encoder");

        let ret = callback(&encoder);

        encoder.end_encoding();

        ret
    }

    fn with_copy_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as crate::backends::common::Backend>::CopyEncoder) -> T,
    ) -> T {
        let encoder = self.new_blit_command_encoder().expect("Failed to create blit command encoder");

        let ret = callback(&encoder);

        encoder.end_encoding();

        ret
    }

    fn submit(&self) {
        self.commit();
    }

    fn wait_until_completed(&self) {
        self.deref().wait_until_completed();
    }

    fn gpu_execution_time_ms(&self) -> Option<f64> {
        match (self.kernel_start_time(), self.kernel_end_time()) {
            (Some(start), Some(end)) => Some((end - start) * 1000.0),
            _ => None,
        }
    }
}

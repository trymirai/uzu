use metal::{MTLCommandBuffer, MTLCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::common::CommandBuffer;

use super::Metal;

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
}

use metal::{MTLBlitCommandEncoder, MTLBuffer};
use objc2::runtime::ProtocolObject;

use crate::backends::common::CopyEncoder;

use super::Metal;

impl CopyEncoder for ProtocolObject<dyn MTLBlitCommandEncoder> {
    type Backend = Metal;

    fn encode_copy(
        &self,
        src: &<Self::Backend as crate::backends::common::Backend>::NativeBuffer,
        dst: &<Self::Backend as crate::backends::common::Backend>::NativeBuffer,
        size: usize,
    ) {
        assert!(src.length() <= size && dst.length() <= size);

        self.copy_buffer_to_buffer(src, 0, dst, 0, size);
    }
}

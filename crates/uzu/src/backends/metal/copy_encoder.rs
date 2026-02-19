use std::ops::Range;

use metal::{MTLBlitCommandEncoder, MTLBlitCommandEncoderExt, MTLBuffer};
use objc2::runtime::ProtocolObject;

use super::Metal;
use crate::backends::common::{Backend, CopyEncoder};

impl CopyEncoder for ProtocolObject<dyn MTLBlitCommandEncoder> {
    type Backend = Metal;

    fn encode_copy(
        &self,
        src: &<Self::Backend as Backend>::NativeBuffer,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        size: usize,
    ) {
        assert!(src.length() >= size && dst.length() >= size);

        self.copy_buffer_to_buffer(src, 0, dst, 0, size);
    }

    fn encode_fill(
        &self,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        range: Range<usize>,
        value: u8,
    ) {
        assert!(range.end > range.start && range.end <= dst.length());
        assert!(range.start % 4 == 0 && range.end % 4 == 0);

        self.fill_buffer_range_value(dst, range, value);
    }
}

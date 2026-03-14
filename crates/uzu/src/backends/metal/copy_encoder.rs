use std::ops::Range;

use metal::{MTLBlitCommandEncoder, MTLBlitCommandEncoderExt, MTLBuffer};
use objc2::runtime::ProtocolObject;

use super::Metal;
use crate::backends::common::{Backend, CopyEncoder};

impl CopyEncoder for ProtocolObject<dyn MTLBlitCommandEncoder> {
    type Backend = Metal;

    fn encode_copy_ranges(
        &self,
        src: (&<Self::Backend as Backend>::Buffer, usize),
        dst: (&<Self::Backend as Backend>::Buffer, usize),
        size: usize,
    ) {
        let (src_buffer, src_offset) = src;
        let (dst_buffer, dst_offset) = dst;
        assert!(src_buffer.length() >= src_offset + size && dst_buffer.length() >= dst_offset + size);

        self.copy_buffer_to_buffer(src_buffer, src_offset, dst_buffer, dst_offset, size);
    }

    fn encode_fill(
        &self,
        dst: &<Self::Backend as Backend>::Buffer,
        range: Range<usize>,
        value: u8,
    ) {
        assert!(range.end > range.start && range.end <= dst.length());
        assert!(range.start % 4 == 0 && range.end % 4 == 0);

        self.fill_buffer_range_value(dst, range, value);
    }
}

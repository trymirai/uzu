use std::ops::Range;

use super::Backend;

pub trait CopyEncoder {
    type Backend: Backend;

    fn encode_copy_ranges(
        &self,
        src: (&<Self::Backend as Backend>::NativeBuffer, usize),
        dst: (&<Self::Backend as Backend>::NativeBuffer, usize),
        size: usize,
    );

    fn encode_copy(
        &self,
        src: &<Self::Backend as Backend>::NativeBuffer,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        size: usize,
    ) {
        self.encode_copy_ranges((src, 0), (dst, 0), size);
    }

    fn encode_fill(
        &self,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        range: Range<usize>,
        value: u8,
    );
}

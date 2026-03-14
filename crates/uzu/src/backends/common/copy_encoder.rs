use std::ops::Range;

use super::Backend;

pub trait CopyEncoder {
    type Backend: Backend;

    fn encode_copy_ranges(
        &self,
        src: (&<Self::Backend as Backend>::Buffer, usize),
        dst: (&<Self::Backend as Backend>::Buffer, usize),
        size: usize,
    );

    fn encode_copy(
        &self,
        src: &<Self::Backend as Backend>::Buffer,
        dst: &<Self::Backend as Backend>::Buffer,
        size: usize,
    ) {
        self.encode_copy_ranges((src, 0), (dst, 0), size);
    }

    fn encode_fill(
        &self,
        dst: &<Self::Backend as Backend>::Buffer,
        range: Range<usize>,
        value: u8,
    );
}

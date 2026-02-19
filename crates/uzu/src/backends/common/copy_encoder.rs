use std::ops::Range;

use super::Backend;

pub trait CopyEncoder {
    type Backend: Backend;

    fn encode_copy(
        &self,
        src: &<Self::Backend as Backend>::NativeBuffer,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        size: usize,
    );

    fn encode_fill(
        &self,
        dst: &<Self::Backend as Backend>::NativeBuffer,
        range: Range<usize>,
        value: u8,
    );
}

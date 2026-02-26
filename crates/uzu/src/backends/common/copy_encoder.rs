use std::ops::Range;

use super::Backend;

pub trait CopyEncoder {
    type Backend: Backend;

    fn encode_copy(
        &mut self,
        src: &<Self::Backend as Backend>::NativeBuffer,
        dst: &mut <Self::Backend as Backend>::NativeBuffer,
        size: usize,
    );

    fn encode_fill(
        &mut self,
        dst: &mut <Self::Backend as Backend>::NativeBuffer,
        range: Range<usize>,
        value: u8,
    );
}

use bytemuck::fill_zeroes;

use super::kv_cache_layer::ArrayCell;
use crate::device::array::Array;

#[derive(Debug)]
pub struct ShortConvLayer {
    pub conv_state: ArrayCell,
}

impl ShortConvLayer {
    pub fn zero(&self) {
        let mut conv = self.conv_state.borrow_mut();
        fill_zeroes(conv.buffer_mut());
    }
}

use bytemuck::fill_zeroes;

use super::kv_cache_layer::ArrayCell;

#[derive(Debug)]
pub struct ShortConvLayer {
    pub conv_state: ArrayCell,
}

impl ShortConvLayer {
    pub fn zero(&self) {
        let mut conv = self.conv_state.borrow_mut();
        fill_zeroes(conv.as_bytes_mut());
    }
}

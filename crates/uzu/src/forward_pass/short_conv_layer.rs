use bytemuck::fill_zeroes;

use super::kv_cache_layer::ArrayCell;
use crate::{Array, DeviceContext};

#[derive(Debug)]
pub struct ShortConvLayer<C: DeviceContext> {
    pub conv_state: ArrayCell<C>,
}

impl<C: DeviceContext> ShortConvLayer<C> {
    pub fn zero(&self) {
        let mut conv = self.conv_state.borrow_mut();
        fill_zeroes(conv.buffer_mut());
    }
}

use bytemuck::fill_zeroes;

use super::kv_cache_layer::ArrayCell;
use crate::{Array, DeviceContext};

#[derive(Debug)]
pub struct SSMLayer<C: DeviceContext> {
    pub conv_state: ArrayCell<C>,
    pub ssm_state: ArrayCell<C>,
}

impl<C: DeviceContext> SSMLayer<C> {
    pub fn zero(&self) {
        {
            let mut conv = self.conv_state.borrow_mut();
            fill_zeroes(conv.buffer_mut());
        }
        {
            let mut ssm = self.ssm_state.borrow_mut();
            fill_zeroes(ssm.buffer_mut());
        }
    }
}

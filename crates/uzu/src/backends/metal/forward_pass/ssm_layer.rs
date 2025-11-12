use bytemuck::fill_zeroes;

use super::kv_cache_layer::ArrayCell;
use crate::device::array::Array;

#[derive(Debug)]
pub struct SSMLayer {
    pub conv_state: ArrayCell,
    pub ssm_state: ArrayCell,
}

impl SSMLayer {
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

use bytemuck::fill_zeroes;

use super::kv_cache_layer::ArrayCell;

#[derive(Debug)]
pub struct SSMLayer {
    pub conv_state: ArrayCell,
    pub ssm_state: ArrayCell,
}

impl SSMLayer {
    pub fn zero(&self) {
        {
            let mut conv = self.conv_state.borrow_mut();
            fill_zeroes(conv.as_bytes_mut());
        }
        {
            let mut ssm = self.ssm_state.borrow_mut();
            fill_zeroes(ssm.as_bytes_mut());
        }
    }
}

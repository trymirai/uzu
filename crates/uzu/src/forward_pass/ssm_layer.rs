use bytemuck::fill_zeroes;

use crate::{array::ArrayCell, backends::common::Backend};

#[derive(Debug)]
pub struct SSMLayer<B: Backend> {
    pub conv_state: ArrayCell<B>,
    pub ssm_state: ArrayCell<B>,
}

impl<B: Backend> SSMLayer<B> {
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

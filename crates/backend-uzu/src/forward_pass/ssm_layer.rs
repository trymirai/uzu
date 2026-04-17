use bytemuck::fill_zeroes;

use crate::{array::Array, backends::common::Backend};

#[derive(Debug)]
pub struct SSMLayer<B: Backend> {
    pub conv_state: Array<B>,
    pub ssm_state: Array<B>,
}

impl<B: Backend> SSMLayer<B> {
    pub fn zero(&mut self) {
        fill_zeroes(self.conv_state.as_bytes_mut());
        fill_zeroes(self.ssm_state.as_bytes_mut());
    }
}

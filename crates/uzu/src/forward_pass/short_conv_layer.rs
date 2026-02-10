use bytemuck::fill_zeroes;

use crate::{array::ArrayCell, backends::common::Backend};

#[derive(Debug)]
pub struct ShortConvLayer<B: Backend> {
    pub conv_state: ArrayCell<B>,
}

impl<B: Backend> ShortConvLayer<B> {
    pub fn zero(&self) {
        let mut conv = self.conv_state.borrow_mut();
        fill_zeroes(conv.as_bytes_mut());
    }
}

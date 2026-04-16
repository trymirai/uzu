use crate::{
    DataType,
    backends::common::{Allocation, Backend, allocation_helpers},
};

pub struct DeltaNetLayer<B: Backend> {
    pub conv_state: Allocation<B>,
    pub conv_shape: [usize; 2],
    pub ssm_state: Allocation<B>,
    pub ssm_shape: [usize; 3],
    pub data_type: DataType,
}

impl<B: Backend> DeltaNetLayer<B> {
    pub fn zero(&mut self) {
        allocation_helpers::fill_allocation(&mut self.conv_state, 0);
        allocation_helpers::fill_allocation(&mut self.ssm_state, 0);
    }
}

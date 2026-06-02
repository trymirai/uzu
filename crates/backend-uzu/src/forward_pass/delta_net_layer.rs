use crate::{
    backends::common::{Allocation, Backend},
    data_type::DataType,
};

pub struct DeltaNetLayer<B: Backend> {
    pub conv_state: Allocation<B>,
    pub conv_shape: [usize; 2],
    pub ssm_state: Allocation<B>,
    pub ssm_shape: [usize; 3],
    pub data_type: DataType,
}

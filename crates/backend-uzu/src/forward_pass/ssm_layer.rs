use crate::{
    backends::common::{Allocation, Backend},
    data_type::DataType,
};

pub struct SSMLayer<B: Backend> {
    pub conv_state: Option<Allocation<B>>,
    pub conv_shape: [usize; 2],
    pub ssm_state: Allocation<B>,
    pub ssm_shape: [usize; 3],
    pub data_type: DataType,
}

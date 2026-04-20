use crate::{
    DataType,
    backends::common::{Allocation, Backend},
};

pub struct DeltaNetLayer<B: Backend> {
    pub conv_state: Allocation<B>,
    pub conv_shape: [usize; 2],
    pub ssm_state: Allocation<B>,
    pub ssm_shape: [usize; 3],
    pub data_type: DataType,
}

use crate::backends::common::kernel::ActivationQuantization;

#[derive(Clone)]
pub struct PreparedInt8A {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub group_sums: Vec<i32>,
    pub quantization: ActivationQuantization,
}

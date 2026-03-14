use crate::backends::common::Backend;

use super::QuantizedMatmulType;

pub struct QuantizedMatmulArguments<'a, B: Backend> {
    pub a_buffer: &'a B::Buffer,
    pub a_offset: usize,
    pub b_buffer: &'a B::Buffer,
    pub scales_buffer: &'a B::Buffer,
    pub zero_points_or_biases_buffer: &'a B::Buffer,
    pub output_buffer: &'a mut B::Buffer,
    pub batch: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub quantization_type: QuantizedMatmulType,
}

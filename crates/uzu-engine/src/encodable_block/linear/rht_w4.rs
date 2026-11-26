use crate::{
    backends::common::{
        Backend, Buffer,
        gpu_types::{hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE, quantization::QuantizationMode},
        kernel::matmul::{MatmulB, QuantizedCorrection, WeightsLayout},
    },
    encodable_block::weight_matrix::WeightMatrix,
};

/// A mirai W4 projection (asymmetric, group 32, no biases) with RHT signs on its input and output.
pub struct RhtW4Layer<'a, B: Backend> {
    pub matrix: &'a WeightMatrix<B>,
    pub scales: &'a B::GlobalGpuBuffer,
    pub zero_points: &'a B::GlobalGpuBuffer,
    pub input_signs: &'a B::GlobalGpuBuffer,
    pub output_signs: &'a B::GlobalGpuBuffer,
    /// Elements between consecutive groups of the scales (the padded output dimension).
    pub scale_stride: u32,
    pub weights_layout: WeightsLayout<B>,
}

impl<'a, B: Backend> RhtW4Layer<'a, B> {
    pub fn new(
        matrix: &'a WeightMatrix<B>,
        has_biases: bool,
        input_signs: &'a B::GlobalGpuBuffer,
        output_signs: &'a B::GlobalGpuBuffer,
    ) -> Option<Self> {
        if !input_signs.size().is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE as usize * size_of::<i32>())
            || !output_signs.size().is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE as usize * size_of::<i32>())
        {
            return None;
        }
        let MatmulB::Quantized(quantized) = matrix.matmul_b() else {
            return None;
        };
        let w4 = quantized.mode == QuantizationMode::U4
            && matches!(quantized.correction, QuantizedCorrection::ZeroPoints(_))
            && quantized.group_size == 32
            && !quantized.signed_codes
            && !has_biases;
        if !w4 {
            return None;
        }
        let QuantizedCorrection::ZeroPoints(zero_points) = quantized.correction else {
            unreachable!("W4 requires zero-point correction")
        };
        Some(Self {
            matrix,
            scales: quantized.scales,
            zero_points,
            input_signs,
            output_signs,
            scale_stride: quantized.params.scale_shape()[1],
            weights_layout: quantized.weights_layout,
        })
    }
}

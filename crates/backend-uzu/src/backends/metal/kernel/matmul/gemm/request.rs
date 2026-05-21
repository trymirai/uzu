use crate::backends::{
    common::{
        Allocation,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::matmul::MatmulArguments,
    },
    metal::{Metal, kernel::TensorAddBiasMetalKernel},
};

pub enum GemmRequest<'a> {
    Fp {
        bias_add: &'a mut TensorAddBiasMetalKernel,
        arguments: MatmulArguments<'a, Metal>,
        use_mxu: bool,
    },
    Quant {
        method: QuantizationMethod,
        mode: QuantizationMode,
        group_size: u32,
        a: &'a Allocation<Metal>,
        a_offset: usize,
        b: &'a Allocation<Metal>,
        scales: &'a Allocation<Metal>,
        zero_points_or_biases: &'a Allocation<Metal>,
        d: &'a mut Allocation<Metal>,
        batch_dim: u32,
        input_dim: u32,
        output_dim: u32,
    },
}

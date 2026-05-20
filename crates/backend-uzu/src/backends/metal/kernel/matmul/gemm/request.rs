use crate::backends::{
    common::kernel::{
        matmul::MatmulArguments,
        quant_matmul::{QuantizedMatmulArguments, QuantizedMatmulConfiguration},
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
        configuration: &'a QuantizedMatmulConfiguration,
        arguments: QuantizedMatmulArguments<'a, Metal>,
    },
}

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{
    ArrayElement,
    backends::{common::gpu_types::QuantizationMethod, cpu::kernel::matmul::gemv::qmv::qmv},
};

#[kernel(QuantizedMatmulQmvFast)]
#[variants(WeightT, f32, f16, bf16)]
#[variants(InputT, f32, f16, bf16)]
#[variants(OutputT, f32, f16, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4, 8)]
pub fn quantized_matmul_qmv_fast<
    WeightT: ArrayElement + Float,
    InputT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    const GROUP_SIZE: u32,
    const BITS: u32,
>(
    weights: *const u32,
    scales: *const WeightT,
    #[optional(quant_method == QuantizationMethod::ScaleZeroPoint)] zero_points: Option<*const u8>,
    #[optional(quant_method == QuantizationMethod::ScaleBias)] biases: Option<*const WeightT>,
    input: *const InputT,
    output: *mut OutputT,
    #[optional(use_hadamard)] hadamard_factors: Option<*const i32>,
    in_vec_size: u32,
    out_vec_size: u32,
    batch_size: u32,
    #[specialize] quant_method: QuantizationMethod,
    #[specialize] use_hadamard: bool,
) {
    if use_hadamard {
        unimplemented!("not supported yet");
    }
    qmv::<WeightT, InputT, OutputT>(
        weights,
        scales,
        zero_points,
        biases,
        input,
        output,
        in_vec_size as usize,
        out_vec_size as usize,
        batch_size as usize,
        quant_method,
        GROUP_SIZE as usize,
        BITS as usize,
    );
}

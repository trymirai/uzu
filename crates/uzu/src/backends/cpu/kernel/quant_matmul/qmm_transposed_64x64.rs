use dsl::kernel;
use half::bf16;
use num_traits::Float;

use crate::{ArrayElement, backends::cpu::kernel::quant_matmul::qmm_transposed::qmm_transposed};

#[kernel(QuantizedMatmulQmmTransposed64x64)]
#[variants(T, bf16)]
#[variants(GROUP_SIZE, 64, 128)]
#[variants(BITS, 4, 8)]
pub fn quantized_matmul_qmm_transposed64x64<T: ArrayElement + Float, const GROUP_SIZE: u32, const BITS: u32>(
    weights: *const u32,
    scales: *const T,
    #[optional(use_zero_points)] zero_points: Option<*const u8>,
    #[optional(use_mlx_quant)] biases: Option<*const T>,
    input: *const T,
    output: *mut T,
    in_vec_size: u32,
    out_vec_size: u32,
    batch_size: u32,
    #[specialize] use_zero_points: bool,
    #[specialize] use_mlx_quant: bool,
) {
    qmm_transposed::<T>(
        weights,
        scales,
        zero_points,
        biases,
        input,
        output,
        in_vec_size as usize,
        out_vec_size as usize,
        batch_size as usize,
        use_zero_points,
        use_mlx_quant,
        GROUP_SIZE as usize,
        BITS as usize,
    );
}

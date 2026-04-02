use dsl::kernel;
use half::bf16;
use num_traits::Float;

use crate::{ArrayElement, backends::cpu::kernel::quant_matmul::qmm_transposed::qmm_transposed};

#[kernel(QuantizedMatmulQmmTransposedWide)]
#[variants(T, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4, 8)]
pub fn quantized_matmul_qmm_transposed_wide<T: ArrayElement + Float, const GROUP_SIZE: i32, const BITS: i32>(
    w: *const u32,
    scales: *const T,
    #[optional(use_zero_points)] zero_points: Option<*const u8>,
    #[optional(use_mlx_quant)] biases: Option<*const T>,
    x: *const T,
    y: *mut T,
    k: i32,
    n: i32,
    m: i32,
    #[specialize] use_zero_points: bool,
    #[specialize] use_mlx_quant: bool,
) {
    qmm_transposed::<T>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        k as usize,
        n as usize,
        m as usize,
        use_zero_points,
        use_mlx_quant,
        GROUP_SIZE as usize,
        BITS as usize,
    );
}

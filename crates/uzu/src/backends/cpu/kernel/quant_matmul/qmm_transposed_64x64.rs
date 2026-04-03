use dsl::kernel;
use half::bf16;
use num_traits::Float;

use crate::ArrayElement;

#[kernel(QuantizedMatmulQmmTransposed64x64)]
#[variants(T, bf16)]
#[variants(GROUP_SIZE, 64, 128)]
#[variants(BITS, 4, 8)]
pub fn quantized_matmul_qmm_transposed64x64<T: ArrayElement + Float, const GROUP_SIZE: u32, const BITS: u32>(
    #[allow(unused)] weights: *const u32,
    #[allow(unused)] scales: *const T,
    #[allow(unused)]
    #[optional(use_zero_points)]
    zero_points: Option<*const u8>,
    #[allow(unused)]
    #[optional(use_mlx_quant)]
    biases: Option<*const T>,
    #[allow(unused)] input: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] in_vec_size: u32,
    #[allow(unused)] out_vec_size: u32,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)]
    #[specialize]
    use_zero_points: bool,
    #[allow(unused)]
    #[specialize]
    use_mlx_quant: bool,
) {
    todo!()
}

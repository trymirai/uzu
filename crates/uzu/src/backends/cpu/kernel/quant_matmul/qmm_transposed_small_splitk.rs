use dsl::kernel;
use half::bf16;
use num_traits::Float;

use crate::ArrayElement;

#[kernel(QuantizedMatmulQmmTransposedSmallSplitKPartial)]
#[variants(T, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4, 8)]
pub fn quantized_matmul_qmm_transposed_small_splitk_partial<
    T: ArrayElement + Float,
    const GROUP_SIZE: i32,
    const BITS: i32,
>(
    #[allow(unused)] w: *const u32,
    #[allow(unused)] scales: *const T,
    #[allow(unused)]
    #[optional(use_zero_points)]
    zero_points: Option<*const u8>,
    #[allow(unused)]
    #[optional(use_mlx_quant)]
    biases: Option<*const T>,
    #[allow(unused)] x: *const T,
    #[allow(unused)] partial: *mut T,
    #[allow(unused)] k: i32,
    #[allow(unused)] n: i32,
    #[allow(unused)] m: i32,
    #[allow(unused)] split_k: i32,
    #[allow(unused)]
    #[specialize]
    use_zero_points: bool,
    #[allow(unused)]
    #[specialize]
    use_mlx_quant: bool,
) {
    todo!()
}

#[kernel(QuantizedMatmulQmmTransposedSmallSplitKReduce)]
#[variants(T, bf16)]
pub fn quantized_matmul_qmm_transposed_small_splitk_reduce<T: ArrayElement + Float>(
    #[allow(unused)] partial: *const T,
    #[allow(unused)] y: *mut T,
    #[allow(unused)] n: i32,
    #[allow(unused)] m: i32,
    #[allow(unused)] split_k: i32,
) {
    todo!()
}

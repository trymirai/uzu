use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use super::qmv_fast_lloyd_max_merged::quantized_matmul_qmv_lloyd_max;
use crate::array::ArrayElement;

#[kernel(QuantizedMatmulQmvLloydMax)]
#[variants(T, f32, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4)]
pub fn quantized_matmul_qmv_lloyd_max_kernel<T: ArrayElement + Float, const GROUP_SIZE: u32, const BITS: u32>(
    weights: *const u32,
    scales: *const T,
    codebook: *const f16,
    bias_indices: *const u8,
    bias_codebook: *const f16,
    input: *const T,
    output: *mut T,
    in_vec_size: u32,
    out_vec_size: u32,
    batch_size: u32,
) {
    let _ = BITS;
    quantized_matmul_qmv_lloyd_max::<T, GROUP_SIZE>(
        weights,
        scales,
        codebook,
        bias_indices,
        bias_codebook,
        input,
        output,
        in_vec_size as usize,
        out_vec_size as usize,
        batch_size as usize,
    );
}

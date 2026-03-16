use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioQuantizerDecode)]
#[variants(T, f32, f16, bf16)]
pub fn audio_quantizer_decode<T: ArrayElement + Float>(
    #[allow(unused)] tokens: *const u32,
    #[allow(unused)] lengths: *const i32,
    #[allow(unused)] semantic_codebook: *const T,
    #[allow(unused)] semantic_out_proj: *const T,
    #[allow(unused)] semantic_out_bias: *const T,
    #[allow(unused)] residual_codebooks: *const T,
    #[allow(unused)] residual_out_proj: *const T,
    #[allow(unused)] residual_out_bias: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] batch_size: i32,
    #[allow(unused)] total_codebooks: i32,
    #[allow(unused)] seq_len: i32,
    #[allow(unused)] input_dim: i32,
    #[allow(unused)] codebook_dim: i32,
    #[allow(unused)] residual_quantizers: i32,
    #[allow(unused)] semantic_cardinality: i32,
    #[allow(unused)] residual_cardinality: i32,
) {
    todo!()
}

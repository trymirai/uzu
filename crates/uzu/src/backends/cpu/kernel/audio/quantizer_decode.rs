use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioQuantizerDecode)]
#[variants(T, f32, f16, bf16)]
pub fn audio_quantizer_decode<T: ArrayElement + Float>(
    tokens: *const u32,
    lengths: *const i32,
    semantic_codebook: *const T,
    semantic_out_proj: *const T,
    semantic_out_bias: *const T,
    residual_codebooks: *const T,
    residual_out_proj: *const T,
    residual_out_bias: *const T,
    output: *mut T,
    batch_size: i32,
    total_codebooks: i32,
    seq_len: i32,
    input_dim: i32,
    codebook_dim: i32,
    residual_quantizers: i32,
    semantic_cardinality: i32,
    residual_cardinality: i32,
) {
    let _ = (
        tokens,
        lengths,
        semantic_codebook,
        semantic_out_proj,
        semantic_out_bias,
        residual_codebooks,
        residual_out_proj,
        residual_out_bias,
        output,
        batch_size,
        total_codebooks,
        seq_len,
        input_dim,
        codebook_dim,
        residual_quantizers,
        semantic_cardinality,
        residual_cardinality,
    );
    todo!()
}

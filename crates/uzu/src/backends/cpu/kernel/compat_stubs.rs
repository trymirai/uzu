use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioCausalConv1dGrouped)]
#[variants(T, f32, f16, bf16)]
pub fn audio_causal_conv1d_grouped<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] weight: *const T,
    #[allow(unused)] bias: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] lengths: *const i32,
    #[allow(unused)] cin: i32,
    #[allow(unused)] cout: i32,
    #[allow(unused)] seq_len: i32,
    #[allow(unused)] kernel_size: i32,
    #[allow(unused)] dilation: i32,
    #[allow(unused)] groups: i32,
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

#[kernel(AudioCausalConv1dGroupedResidual)]
#[variants(T, f32, f16, bf16)]
pub fn audio_causal_conv1d_grouped_residual<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] residual: *const T,
    #[allow(unused)] weight: *const T,
    #[allow(unused)] bias: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] lengths: *const i32,
    #[allow(unused)] cin: i32,
    #[allow(unused)] cout: i32,
    #[allow(unused)] seq_len: i32,
    #[allow(unused)] kernel_size: i32,
    #[allow(unused)] dilation: i32,
    #[allow(unused)] groups: i32,
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

#[kernel(AudioCausalConvTranspose1dCausalPad)]
#[variants(T, f32, f16, bf16)]
pub fn audio_causal_conv_transpose1d_causal_pad<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] weight: *const T,
    #[allow(unused)] bias: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] lengths: *const i32,
    #[allow(unused)] cin: i32,
    #[allow(unused)] cout: i32,
    #[allow(unused)] seq_len_in: i32,
    #[allow(unused)] seq_len_out: i32,
    #[allow(unused)] kernel_size: i32,
    #[allow(unused)] stride: i32,
    #[allow(unused)] groups: i32,
    #[allow(unused)] input_layout: i32,
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

#[kernel(AudioNormNcs)]
#[variants(T, f32, f16, bf16)]
pub fn audio_norm_ncs<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] scales: *const T,
    #[allow(unused)] bias: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] lengths: *const i32,
    #[allow(unused)] channels: i32,
    #[allow(unused)] seq_len: i32,
    #[allow(unused)] epsilon: f32,
    #[allow(unused)] subtract_mean: i32,
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

#[kernel(AudioQuantizerDecode)]
#[variants(T, f32, f16, bf16)]
pub fn audio_quantizer_decode<T: ArrayElement + Float>(
    #[allow(unused)] tokens: *const i32,
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

#[kernel(AudioTransposeNscToNcs)]
#[variants(T, f32, f16, bf16)]
pub fn audio_transpose_nsc_to_ncs<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] seq_len: i32,
    #[allow(unused)] channels: i32,
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

#[kernel(EmbeddingRowsSum)]
#[variants(T, f32, f16, bf16)]
pub fn embedding_rows_sum<T: ArrayElement + Float>(
    #[allow(unused)] row_indices: *const u64,
    #[allow(unused)] weights: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] num_rows: u32,
    #[allow(unused)] total_rows: u32,
    #[allow(unused)] model_dim: u32,
) {
    todo!()
}

#[kernel(RepetitionPenalty)]
#[variants(T, f32, f16, bf16)]
pub fn repetition_penalty<T: ArrayElement + Float>(
    #[allow(unused)] logits: *mut T,
    #[allow(unused)] previous_tokens: *const u32,
    #[allow(unused)] previous_counts: *const u32,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] vocab_size: u32,
    #[allow(unused)] max_previous_tokens: u32,
    #[allow(unused)] repetition_penalty: f32,
) {
    todo!()
}

#[kernel(TensorAddScale)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_add_scale<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] bias: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] num_cols: u32,
    #[allow(unused)] length: u32,
    #[allow(unused)] scale: f32,
) {
    todo!()
}

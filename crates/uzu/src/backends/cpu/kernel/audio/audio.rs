use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioFsqDecode)]
#[variants(T, f32, f16, bf16)]
pub fn audio_fsq_decode<T: ArrayElement + Float>(
    #[allow(unused)] tokens: *const i32,
    #[allow(unused)] out: *mut T,
    #[allow(unused)] lengths: *const i32,
    #[allow(unused)] num_groups: i32,
    #[allow(unused)] seq_len: i32,
    #[allow(unused)] codebook_dim: i32,
    #[allow(unused)] num_levels: &[i32],
    #[allow(unused)] dim_base_index: &[i32],
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

#[kernel(AudioAdd)]
#[variants(T, f32, f16, bf16)]
pub fn audio_add<T: ArrayElement + Float>(
    #[allow(unused)] a: *const T,
    #[allow(unused)] b: *const T,
    #[allow(unused)] out: *mut T,
    #[allow(unused)] n: i32,
) {
    todo!()
}

#[kernel(AudioCausalConv1d)]
#[variants(T, f32, f16, bf16)]
pub fn audio_causal_conv1d<T: ArrayElement + Float>(
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
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

#[kernel(AudioCausalConvTranspose1d)]
#[variants(T, f32, f16, bf16)]
pub fn audio_causal_conv_transpose1d<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] weight: *const T,
    #[allow(unused)] bias: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] lengths: *const i32,
    #[allow(unused)] cin: i32,
    #[allow(unused)] cout: i32,
    #[allow(unused)] seq_len_in: i32,
    #[allow(unused)] seq_len_out: i32,
    #[allow(unused)] stride: i32,
    #[allow(unused)] groups: i32,
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

#[kernel(AudioHalfSnake)]
#[variants(T, f32, f16, bf16)]
pub fn audio_half_snake<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] alpha: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] channels: i32,
    #[allow(unused)] seq_len: i32,
    #[allow(unused)] snake_channels: i32,
    #[allow(unused)] negative_slope: f32,
    #[allow(unused)] eps: f32,
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

#[kernel(AudioClamp)]
#[variants(T, f32, f16, bf16)]
pub fn audio_clamp<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] n: i32,
    #[allow(unused)] min_value: f32,
    #[allow(unused)] max_value: f32,
) {
    todo!()
}

#[kernel(AudioConv1d)]
#[variants(T, f32, f16, bf16)]
pub fn audio_conv1d<T: ArrayElement + Float>(
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
    #[allow(unused)] dilation: i32,
    #[allow(unused)] padding: i32,
    #[allow(unused)] pad_mode: i32,
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

#[kernel(AudioFsqEncode)]
#[variants(T, f32, f16, bf16)]
pub fn audio_fsq_encode<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] tokens: *mut i32,
    #[allow(unused)] lengths: *const i32,
    #[allow(unused)] num_groups: i32,
    #[allow(unused)] seq_len: i32,
    #[allow(unused)] codebook_dim: i32,
    #[allow(unused)] num_levels: &[i32],
    #[allow(unused)] dim_base_index: &[i32],
    #[allow(unused)] eps: f32,
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}

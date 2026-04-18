use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioNormNcs)]
#[variants(T, f32, f16, bf16)]
pub fn audio_norm_ncs<T: ArrayElement + Float>(
    input: *const T,
    scales: *const T,
    bias: *const T,
    output: *mut T,
    lengths: *const i32,
    channels: i32,
    seq_len: i32,
    epsilon: f32,
    subtract_mean: i32,
    batch_size: i32,
) {
    let _ = (input, scales, bias, output, lengths, channels, seq_len, epsilon, subtract_mean, batch_size);
    todo!()
}

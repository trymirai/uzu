use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioHalfSnake)]
#[variants(T, f32, f16, bf16)]
pub fn audio_half_snake<T: ArrayElement + Float>(
    input: *const T,
    alpha: *const T,
    output: *mut T,
    channels: i32,
    seq_len: i32,
    snake_channels: i32,
    negative_slope: f32,
    eps: f32,
    batch_size: i32,
) {
    let _ = (input, alpha, output, channels, seq_len, snake_channels, negative_slope, eps, batch_size);
    todo!()
}

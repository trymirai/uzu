use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

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

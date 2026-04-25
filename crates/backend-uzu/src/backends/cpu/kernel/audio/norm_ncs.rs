use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

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

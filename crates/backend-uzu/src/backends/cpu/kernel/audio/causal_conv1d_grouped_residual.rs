use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

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

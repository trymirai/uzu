use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioCausalConv1dGroupedResidual)]
#[variants(T, f32, f16, bf16)]
pub fn audio_causal_conv1d_grouped_residual<T: ArrayElement + Float>(
    input: *const T,
    residual: *const T,
    weight: *const T,
    bias: *const T,
    output: *mut T,
    lengths: *const i32,
    cin: i32,
    cout: i32,
    seq_len: i32,
    kernel_size: i32,
    dilation: i32,
    groups: i32,
    batch_size: i32,
) {
    let _ =
        (input, residual, weight, bias, output, lengths, cin, cout, seq_len, kernel_size, dilation, groups, batch_size);
    todo!()
}

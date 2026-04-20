use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioConv1d)]
#[variants(T, f32, f16, bf16)]
pub fn audio_conv1d<T: ArrayElement + Float>(
    input: *const T,
    weight: *const T,
    bias: *const T,
    output: *mut T,
    lengths: *const i32,
    cin: i32,
    cout: i32,
    seq_len_in: i32,
    seq_len_out: i32,
    kernel_size: i32,
    stride: i32,
    dilation: i32,
    padding: i32,
    pad_mode: i32,
    batch_size: i32,
) {
    let _ = (
        input,
        weight,
        bias,
        output,
        lengths,
        cin,
        cout,
        seq_len_in,
        seq_len_out,
        kernel_size,
        stride,
        dilation,
        padding,
        pad_mode,
        batch_size,
    );
    todo!()
}

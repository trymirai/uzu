use dsl::kernel;
use half::{bf16, f16};
use minijinja::filters::length;
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TensorAddBias)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_add_bias<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] bias: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] num_cols: u32,
    #[allow(unused)] length: u32,
) {
    for i in 0usize..(length as usize) {
        let bias_position = (i % num_cols as usize);
        unsafe {
            *output.add(i) = *input.add(i) + *bias.add(bias_position);
        }
    }
}

use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TensorAddBias)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_add_bias<T: ArrayElement + Float>(
    #[optional(!in_place)] input: Option<*const T>,
    bias: *const T,
    output: *mut T,
    num_cols: u32,
    length: u32,
    #[specialize] in_place: bool,
) {
    let _ = in_place;
    for i in 0usize..(length as usize) {
        let bias_position = i % num_cols as usize;
        unsafe {
            if let Some(in_data) = input {
                *output.add(i) = *in_data.add(i) + *bias.add(bias_position);
            } else {
                *output.add(i) = *output.add(i) + *bias.add(bias_position);
            }
        }
    }
}

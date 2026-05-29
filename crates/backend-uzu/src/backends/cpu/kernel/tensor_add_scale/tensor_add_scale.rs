use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::ArrayElement;

#[kernel(TensorAddScale)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_add_scale<T: ArrayElement + Float>(
    #[optional(!in_place)] input: Option<*const T>,
    bias: *const T,
    output: *mut T,
    num_cols: u32,
    length: u32,
    scale: f32,
    #[allow(unused)]
    #[specialize]
    in_place: bool,
) {
    let num_cols = num_cols as usize;
    for position in 0..length as usize {
        let col = position % num_cols;
        unsafe {
            let input_value = match input {
                Some(input) => (*input.add(position)).to_f32().unwrap(),
                None => (*output.add(position)).to_f32().unwrap(),
            };
            let bias_value = (*bias.add(col)).to_f32().unwrap();
            *output.add(position) = T::from((input_value + bias_value) * scale).unwrap();
        }
    }
}

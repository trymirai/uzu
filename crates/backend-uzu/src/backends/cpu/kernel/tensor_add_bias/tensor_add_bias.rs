use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(TensorAddBias)]
#[variants(T, f32, f16, bf16)]
#[variants(BiasT, f32, f16, bf16)]
pub fn tensor_add_bias<T: ArrayElement + Float, BiasT: ArrayElement + Float>(
    #[allow(unused)]
    #[optional(!in_place)]
    input: Option<*const T>,
    #[allow(unused)] bias: *const BiasT,
    #[allow(unused)]
    #[optional(indexed)]
    bias_row_indices: Option<*const u32>,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] num_cols: u32,
    #[allow(unused)] length: u32,
    #[specialize] in_place: bool,
    #[specialize] indexed: bool,
) {
    assert_eq!(input.is_none(), in_place);
    assert_eq!(bias_row_indices.is_some(), indexed);

    for i in 0usize..(length as usize) {
        let column = i % num_cols as usize;
        let bias_position = match bias_row_indices {
            Some(indices) => unsafe { *indices.add(i / num_cols as usize) as usize * num_cols as usize + column },
            None => column,
        };
        unsafe {
            let value: f32 = if let Some(in_data) = input {
                num_traits::cast(*in_data.add(i)).unwrap()
            } else {
                num_traits::cast(*output.add(i)).unwrap()
            };
            let bias_value: f32 = num_traits::cast(*bias.add(bias_position)).unwrap();
            *output.add(i) = num_traits::cast(value + bias_value).unwrap();
        }
    }
}

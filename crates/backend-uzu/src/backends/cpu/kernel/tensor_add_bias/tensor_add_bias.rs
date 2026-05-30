use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::ArrayElement;

#[kernel(TensorAddBias)]
#[variants(T, f32, f16, bf16)]
#[variants(BiasT, f32, f16, bf16)]
pub fn tensor_add_bias<T: ArrayElement + Float, BiasT: ArrayElement + Float>(
    #[allow(unused)]
    #[optional(!in_place)]
    input: Option<*const T>,
    #[allow(unused)] bias: *const BiasT,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] num_cols: u32,
    #[allow(unused)] length: u32,
    #[specialize] in_place: bool,
) {
    assert_eq!(input.is_none(), in_place);

    for i in 0usize..(length as usize) {
        let bias_position = i % num_cols as usize;
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

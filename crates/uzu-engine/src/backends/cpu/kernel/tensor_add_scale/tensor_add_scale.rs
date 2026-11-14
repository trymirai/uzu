use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

#[kernel(TensorAddScale)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_add_scale<T: DataTypeRepr + Float>(
    #[optional(!in_place)] input: Option<*const T>,
    #[optional(has_bias)] bias: Option<*const T>,
    output: *mut T,
    num_cols: u32,
    length: u32,
    scale: f32,
    #[specialize] in_place: bool,
    #[specialize] has_bias: bool,
) {
    assert_eq!(input.is_none(), in_place);
    assert_eq!(bias.is_some(), has_bias);

    let num_cols = num_cols as usize;
    for position in 0..length as usize {
        let col = position % num_cols;
        unsafe {
            let input_value = match input {
                Some(input) => (*input.add(position)).to_f32().unwrap(),
                None => (*output.add(position)).to_f32().unwrap(),
            };
            let bias_value = match bias {
                Some(bias) => (*bias.add(col)).to_f32().unwrap(),
                None => 0.0,
            };
            *output.add(position) = T::from((input_value + bias_value) * scale).unwrap();
        }
    }
}

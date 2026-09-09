use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::array::ArrayElement;

#[kernel(SeparableCausalConv)]
#[variants(T, bf16)]
pub fn separable_causal_conv<T: ArrayElement + Float>(
    input: *const T,
    coefficient_deltas: *const T,
    weights: *const T,
    #[optional(has_bias)] bias: Option<*const T>,
    output: *mut T,
    sequence_length: u32,
    coefficient_row_stride: u32,
    #[specialize] model_dim: u32,
    #[specialize] kernel_size: u32,
    #[specialize] group_size: u32,
    #[specialize] has_bias: bool,
) {
    let sequence_length = sequence_length as usize;
    let coefficient_row_stride = coefficient_row_stride as usize;
    let model_dim = model_dim as usize;
    let kernel_size = kernel_size as usize;
    let group_size = group_size as usize;

    let num_groups = model_dim / group_size;

    for token in 0..sequence_length {
        for channel in 0..model_dim {
            let group = channel / group_size;
            let mut output_value = if has_bias {
                unsafe { (*bias.unwrap().add(channel)).to_f32().unwrap() }
            } else {
                0.0
            };

            for tokens_back in 0..kernel_size.min(token + 1) {
                let input_token = token - tokens_back;
                let input_index = input_token * model_dim + channel;
                let weight_index = channel * kernel_size + (kernel_size - 1 - tokens_back);
                let coefficient_index = token * coefficient_row_stride + tokens_back * num_groups + group;

                let input_value = unsafe { (*input.add(input_index)).to_f32().unwrap() };
                let base_weight = unsafe { (*weights.add(weight_index)).to_f32().unwrap() };
                let coefficient_delta = unsafe { (*coefficient_deltas.add(coefficient_index)).to_f32().unwrap() };

                output_value += input_value * (base_weight + coefficient_delta);
            }

            let output_index = token * model_dim + channel;
            unsafe {
                *output.add(output_index) = T::from(output_value).unwrap();
            }
        }
    }
}

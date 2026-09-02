use half::bf16;
use uzu_engine_macros::kernel;

use crate::array::ArrayElement;

#[kernel(GroupedConvolution)]
#[variants(T, bf16)]
pub fn grouped_convolution<T: ArrayElement>(
    input: *const T,
    coefficients: *const T,
    base_kernel: *const T,
    output: *mut T,
    sequence_length: u32,
    #[specialize] model_dim: u32,
    #[specialize] group_size: u32,
    #[specialize] kernel_size: u32,
) {
    let sequence_length = sequence_length as usize;
    let model_dim = model_dim as usize;
    let group_size = group_size as usize;
    let kernel_size = kernel_size as usize;
    let groups = model_dim / group_size;
    let coefficient_stride = 2 * kernel_size * groups;
    for token in 0..sequence_length {
        for channel in 0..model_dim {
            let group = channel / group_size;
            let mut value = 0.0;
            for tap in 0..kernel_size {
                if token < tap {
                    continue;
                }
                let coefficient = unsafe {
                    (*base_kernel.add(tap * model_dim + channel)).to_f32().unwrap()
                        + (*coefficients.add(token * coefficient_stride + tap * groups + group)).to_f32().unwrap()
                };
                value += coefficient * unsafe { (*input.add((token - tap) * model_dim + channel)).to_f32().unwrap() };
            }
            unsafe { *output.add(token * model_dim + channel) = T::from(value).unwrap() };
        }
    }
}

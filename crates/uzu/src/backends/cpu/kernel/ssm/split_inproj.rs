use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(SplitInProj)]
#[variants(T, f32, f16, bf16)]
pub fn split_in_proj<T: ArrayElement + Float>(
    input: *const T,
    conv_out: *mut T,
    z_out: *mut T,
    dt_out: *mut T,
    z_bias: *const T,
    suffix_length: u32,
    total_dim: u32,
    conv_dim: u32,
    inner_dim: u32,
    num_heads: u32,
) {
    let suffix_length = suffix_length as usize;
    let total_dim = total_dim as usize;
    let conv_dim = conv_dim as usize;
    let inner_dim = inner_dim as usize;
    let num_heads = num_heads as usize;

    unsafe {
        for row in 0..suffix_length {
            for col in 0..total_dim {
                let input_idx = row * total_dim + col;
                if col < conv_dim {
                    let dst = row * conv_dim + col;
                    *conv_out.add(dst) = *input.add(input_idx);
                } else if col < conv_dim + inner_dim {
                    let dst = row * inner_dim + (col - conv_dim);
                    let bias_idx = col - conv_dim;
                    *z_out.add(dst) = *input.add(input_idx) + *z_bias.add(bias_idx);
                } else if col < conv_dim + inner_dim + num_heads {
                    let dst = row * num_heads + (col - conv_dim - inner_dim);
                    *dt_out.add(dst) = *input.add(input_idx);
                }
            }
        }
    }
}

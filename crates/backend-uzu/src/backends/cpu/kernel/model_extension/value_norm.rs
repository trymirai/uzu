use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(ValueNorm)]
#[variants(T, f32, f16, bf16)]
pub fn value_norm<T: ArrayElement + Float>(
    qkv: *mut T,
    batch_size: u32,
    num_heads: u32,
    num_groups: u32,
    head_dim: u32,
    epsilon: f32,
) {
    let batch_size = batch_size as usize;
    let num_heads = num_heads as usize;
    let num_groups = num_groups as usize;
    let head_dim = head_dim as usize;
    let row_stride = (num_heads + 2 * num_groups) * head_dim;
    let values_offset = (num_heads + num_groups) * head_dim;

    for batch in 0..batch_size {
        for group in 0..num_groups {
            let offset = batch * row_stride + values_offset + group * head_dim;
            let mut sum_sq = 0.0f32;
            for dim in 0..head_dim {
                let value = unsafe { (*qkv.add(offset + dim)).to_f32().unwrap() };
                sum_sq += value * value;
            }
            let rms_inv = (sum_sq / head_dim as f32 + epsilon).sqrt().recip();
            for dim in 0..head_dim {
                unsafe {
                    let value = (*qkv.add(offset + dim)).to_f32().unwrap() * rms_inv;
                    *qkv.add(offset + dim) = T::from(value).unwrap();
                }
            }
        }
    }
}

use dsl::kernel;
use half::{bf16, f16};
use num_traits::{Float, ToPrimitive};

use crate::ArrayElement;

#[kernel(PoolingCls)]
#[variants(T, f32, f16, bf16)]
pub fn pooling_cls<T: ArrayElement + Float>(
    input: *const T,
    output: *mut T,
    seq_len: u32,
    hidden_dim: u32,
    batch_size: u32,
) {
    let seq_len = seq_len as usize;
    let hidden_dim = hidden_dim as usize;
    let batch_size = batch_size as usize;

    for dim_idx in 0..hidden_dim {
        for batch_idx in 0..batch_size {
            unsafe {
                *output.add(batch_idx * hidden_dim + dim_idx) = *input.add(batch_idx * hidden_dim * seq_len + dim_idx);
            }
        }
    }
}

#[kernel(PoolingMean)]
#[variants(T, f32, f16, bf16)]
pub fn pooling_mean<T: ArrayElement + Float>(
    input: *const T,
    output: *mut T,
    seq_len: u32,
    hidden_dim: u32,
    batch_size: u32,
) {
    let seq_len = seq_len as usize;
    let hidden_dim = hidden_dim as usize;
    let batch_size = batch_size as usize;

    for dim_idx in 0..hidden_dim {
        for batch_idx in 0..batch_size {
            let mut sum = 0f32;
            for seq_idx in 0..seq_len {
                unsafe {
                    let position = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx;
                    sum += (*input.add(position)).to_f32().unwrap();
                }
            }
            unsafe {
                *output.add(batch_idx * hidden_dim + dim_idx) = T::from(sum / seq_len.to_f32().unwrap()).unwrap();
            }
        }
    }
}

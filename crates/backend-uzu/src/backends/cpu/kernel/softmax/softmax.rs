use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::ArrayElement;

#[kernel(Softmax)]
#[variants(T, f32, f16, bf16)]
pub fn softmax<T: ArrayElement + Float>(
    values: *mut T,
    #[optional(has_sinks)] sinks: Option<*const T>,
    row_length: u32,
    outer_dim: u32,
    batch_dim: u32,
    #[specialize] has_sinks: bool,
) {
    for outer_index in 0..outer_dim {
        for batch_index in 0..batch_dim {
            let row_offset = ((outer_index * batch_dim + batch_index) * row_length) as usize;
            let sink = if has_sinks {
                unsafe { *sinks.unwrap().add(outer_index as usize) }.to_f32().unwrap()
            } else {
                f32::NEG_INFINITY
            };

            let mut max_value = if has_sinks {
                sink
            } else {
                f32::NEG_INFINITY
            };
            for i in 0..row_length as usize {
                max_value = max_value.max(unsafe { *values.add(row_offset + i) }.to_f32().unwrap());
            }

            let mut sum_exp = if has_sinks {
                (sink - max_value).exp()
            } else {
                0.0
            };
            for i in 0..row_length as usize {
                let p = (unsafe { *values.add(row_offset + i) }.to_f32().unwrap() - max_value).exp();
                sum_exp += p;
                unsafe {
                    *values.add(row_offset + i) = T::from(p).unwrap();
                }
            }

            let inv = 1.0 / sum_exp;
            for i in 0..row_length as usize {
                unsafe {
                    let p = *values.add(row_offset + i);
                    *values.add(row_offset + i) = T::from(p.to_f32().unwrap() * inv).unwrap();
                }
            }
        }
    }
}

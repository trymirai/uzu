use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(TensorMulSlice)]
#[variants(T, f32, f16, bf16)]
pub fn tensor_mul_slice<T: ArrayElement + Float>(
    values: *mut T,
    slice_source: *const T,
    suffix_length: u32,
    total_slice_dim: u32,
    slice_dim: u32,
    slice_index: u32,
) {
    let suffix_length = suffix_length as usize;
    let total_slice_dim = total_slice_dim as usize;
    let slice_dim = slice_dim as usize;
    let slice_index = slice_index as usize;

    for token in 0..suffix_length {
        let values_offset = token * slice_dim;
        let slice_offset = token * total_slice_dim + slice_index * slice_dim;
        for dim in 0..slice_dim {
            unsafe {
                *values.add(values_offset + dim) =
                    *values.add(values_offset + dim) * *slice_source.add(slice_offset + dim);
            }
        }
    }
}

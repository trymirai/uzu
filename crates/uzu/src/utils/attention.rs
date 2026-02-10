use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, DataType, array::Array, backends::common::Backend};

pub fn fill_attention_bias(
    dst: &mut Array<impl Backend>,
    suffix_length: usize,
    prefix_segment_length: usize,
    should_be_neg_inf: impl Fn(usize, usize) -> bool,
) {
    let fill_typed_for_dtype = match dst.data_type() {
        DataType::F16 => fill_typed::<f16>,
        DataType::BF16 => fill_typed::<bf16>,
        DataType::F32 => fill_typed::<f32>,
        DataType::F64 => fill_typed::<f64>,
        _ => panic!("Unsupported data type for attention bias fill"),
    };

    fill_typed_for_dtype(dst, suffix_length, prefix_segment_length, should_be_neg_inf);
}

fn fill_typed<T: ArrayElement + Float>(
    dst: &mut Array<impl Backend>,
    suffix_length: usize,
    prefix_segment_length: usize,
    should_be_neg_inf: impl Fn(usize, usize) -> bool,
) {
    let cols = suffix_length + prefix_segment_length;
    let slice = dst.as_slice_mut::<T>();
    for row in 0..suffix_length {
        for col in 0..cols {
            slice[row * cols + col] = if should_be_neg_inf(row, col) {
                T::neg_infinity()
            } else {
                T::zero()
            };
        }
    }
}

use backend_uzu_macros::kernel;
use half::bf16;
use num_traits::{Float, NumCast};

use super::{hadamard_transform, min_max_symmetric_divisor, quantize_symmetric_i8};
use crate::{
    array::ArrayElement,
    backends::common::gpu_types::{ActivationTransformOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
};

pub fn quantize_transformed_row(
    transformed: &[f32],
    activation_scale_group_size: usize,
    sum_group_size: Option<usize>,
    values: &mut [i8],
    scales: &mut [f32],
    mut group_sums: Option<&mut [i32]>,
) {
    assert_eq!(transformed.len(), values.len());
    assert_eq!(scales.len(), transformed.len() / activation_scale_group_size);
    assert!(transformed.len().is_multiple_of(activation_scale_group_size));
    if let Some(group_sums) = group_sums.as_deref() {
        assert_eq!(group_sums.len(), transformed.len() / sum_group_size.expect("correction group"));
    }
    if let Some(group_sums) = group_sums.as_deref_mut() {
        group_sums.fill(0);
    }

    for (scale_group_index, source) in transformed.chunks_exact(activation_scale_group_size).enumerate() {
        let scale = min_max_symmetric_divisor(source);
        scales[scale_group_index] = scale;
        for (index, &value) in source.iter().enumerate() {
            let code = quantize_symmetric_i8(value, scale);
            let absolute_index = scale_group_index * activation_scale_group_size + index;
            values[absolute_index] = code;
            if let Some(group_sums) = group_sums.as_deref_mut() {
                group_sums[absolute_index / sum_group_size.expect("correction group")] += code as i32;
            }
        }
    }
}

#[kernel(ActivationTransform)]
#[variants(T, f32, bf16)]
pub fn activation_transform<T: ArrayElement + Float>(
    #[optional(!in_place)] input: Option<*const T>,
    #[optional(ops == ActivationTransformOp::InputRht || ops == ActivationTransformOp::OutputRht)] fp_out: Option<
        *mut T,
    >,
    #[optional(ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums)]
    q_out: Option<*mut i8>,
    #[optional(ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums)]
    scales_out: Option<*mut f32>,
    #[optional(ops == ActivationTransformOp::QuantizeWithGroupSums)] group_sums_out: Option<*mut i32>,
    rht_factors: *const i32,
    batch_size: u32,
    element_count: u32,
    #[specialize] ops: ActivationTransformOp,
    #[specialize] in_place: bool,
    #[specialize] activation_scale_group_size: u32,
    #[specialize] sum_group_size: u32,
) {
    let input = match in_place {
        true => fp_out.expect("in-place transform requires fp_out"),
        false => input.expect("out-of-place transform requires input"),
    };
    let rows = batch_size as usize;
    let columns = element_count as usize;
    let input_rht = ops != ActivationTransformOp::OutputRht;
    let quantize = matches!(ops, ActivationTransformOp::Quantize | ActivationTransformOp::QuantizeWithGroupSums);

    let mut transformed = vec![0.0f32; columns];
    for row in 0..rows {
        let row_offset = row * columns;
        for stripe_start in (0..columns).step_by(HADAMARD_TRANSFORM_BLOCK_SIZE as usize) {
            let mut stripe = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize];
            for lane in 0..HADAMARD_TRANSFORM_BLOCK_SIZE as usize {
                let index = stripe_start + lane;
                let value: f32 = NumCast::from(unsafe { *input.add(row_offset + index) }).unwrap();
                let factor = unsafe { *rht_factors.add(index) } as f32;
                stripe[lane] = if input_rht {
                    value * factor
                } else {
                    value
                };
            }

            hadamard_transform(&mut stripe);

            for lane in 0..HADAMARD_TRANSFORM_BLOCK_SIZE as usize {
                let index = stripe_start + lane;
                let factor = unsafe { *rht_factors.add(index) } as f32;
                transformed[index] = if input_rht {
                    stripe[lane]
                } else {
                    stripe[lane] * factor
                };
            }
        }

        if quantize {
            let values = unsafe {
                std::slice::from_raw_parts_mut(
                    q_out.expect("quantized transform requires q_out").add(row_offset),
                    columns,
                )
            };
            let scales_per_row = columns / activation_scale_group_size as usize;
            let scales = unsafe {
                std::slice::from_raw_parts_mut(
                    scales_out.expect("quantized transform requires scales_out").add(row * scales_per_row),
                    scales_per_row,
                )
            };
            let sums_per_row = columns / sum_group_size as usize;
            let sums = group_sums_out.map(|group_sums_out| unsafe {
                std::slice::from_raw_parts_mut(group_sums_out.add(row * sums_per_row), sums_per_row)
            });
            quantize_transformed_row(
                &transformed,
                activation_scale_group_size as usize,
                group_sums_out.map(|_| sum_group_size as usize),
                values,
                scales,
                sums,
            );
        } else {
            let fp_out = fp_out.expect("FP transform requires fp_out");
            for index in 0..columns {
                unsafe {
                    *fp_out.add(row_offset + index) = <T as NumCast>::from(transformed[index]).unwrap();
                }
            }
        }
    }
}

use half::bf16;
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use super::{
    super::hadamard_transform::hadamard_transform::hadamard_transform, min_max_symmetric_divisor, quantize_symmetric_i8,
};
use crate::{
    array::ArrayElement,
    backends::common::gpu_types::{ActivationTransformOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
};

#[kernel(ActivationTransform)]
#[variants(T, f32, bf16)]
pub fn activation_transform<T: ArrayElement + Float>(
    input: *const T,
    #[allow(unused)] fp_out: *mut T,
    #[allow(unused)] q_out: *mut i8,
    #[allow(unused)] scales_out: *mut f32,
    #[optional(ops.contains(ActivationTransformOp::GROUP_SUMS))] group_sums_out: Option<*mut i32>,
    rht_factors: *const i32,
    batch_size: u32,
    element_count: u32,
    group_size: u32,
    #[specialize] ops: ActivationTransformOp,
) {
    let ops = ops.validate();
    let rows = batch_size as usize;
    let columns = element_count as usize;
    let group_size = group_size as usize;
    let input_rht = ops.contains(ActivationTransformOp::INPUT_RHT);
    let quantize = ops.contains(ActivationTransformOp::QUANTIZE);
    assert!(columns.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE));
    if quantize {
        assert!(group_size > 0 && columns.is_multiple_of(group_size));
    }

    let groups = columns.div_ceil(group_size.max(1));
    let mut transformed = vec![0.0f32; columns];
    for row in 0..rows {
        let row_offset = row * columns;
        for stripe_start in (0..columns).step_by(HADAMARD_TRANSFORM_BLOCK_SIZE) {
            let mut stripe = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE];
            for lane in 0..HADAMARD_TRANSFORM_BLOCK_SIZE {
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

            for lane in 0..HADAMARD_TRANSFORM_BLOCK_SIZE {
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
            for group in 0..groups {
                let start = group * group_size;
                let end = (start + group_size).min(columns);
                let slice = &transformed[start..end];
                let divisor = min_max_symmetric_divisor(slice);
                unsafe { *scales_out.add(row * groups + group) = divisor };
                let mut group_sum = 0i32;
                for index in start..end {
                    let q = quantize_symmetric_i8(transformed[index], divisor);
                    unsafe { *q_out.add(row * columns + index) = q };
                    group_sum += q as i32;
                }
                if let Some(group_sums_out) = group_sums_out {
                    unsafe { *group_sums_out.add(row * groups + group) = group_sum };
                }
            }
        } else {
            for index in 0..columns {
                unsafe {
                    *fp_out.add(row_offset + index) = <T as NumCast>::from(transformed[index]).unwrap();
                }
            }
        }
    }
}

use half::bf16;
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use super::{hadamard_transform, min_max_symmetric_divisor, quantize_symmetric_i8};
use crate::{
    array::ArrayElement,
    backends::common::gpu_types::{ActivationTransformOp, HADAMARD_TRANSFORM_BLOCK_SIZE},
};

#[kernel(ActivationTransform)]
#[variants(T, f32, bf16)]
pub fn activation_transform<T: ArrayElement + Float>(
    input: *const T,
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
) {
    let rows = batch_size as usize;
    let columns = element_count as usize;
    let input_rht = ops != ActivationTransformOp::OutputRht;
    let quantize = matches!(ops, ActivationTransformOp::Quantize | ActivationTransformOp::QuantizeWithGroupSums);

    let groups = columns / HADAMARD_TRANSFORM_BLOCK_SIZE;
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
            let q_out = q_out.expect("quantized transform requires q_out");
            let scales_out = scales_out.expect("quantized transform requires scales_out");
            for group in 0..groups {
                let start = group * HADAMARD_TRANSFORM_BLOCK_SIZE;
                let end = start + HADAMARD_TRANSFORM_BLOCK_SIZE;
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
            let fp_out = fp_out.expect("FP transform requires fp_out");
            for index in 0..columns {
                unsafe {
                    *fp_out.add(row_offset + index) = <T as NumCast>::from(transformed[index]).unwrap();
                }
            }
        }
    }
}

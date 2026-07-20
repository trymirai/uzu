use half::{bf16, f16};
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::common::gpu_types::{ACTIVATION_QUANTIZATION_GROUP_SIZE, ActivationPrepareOps, HADAMARD_TRANSFORM_BLOCK_SIZE},
};

use super::{min_max_symmetric_divisor, quantize_symmetric_i8};

fn input_rht(values: &mut [f32; HADAMARD_TRANSFORM_BLOCK_SIZE]) {
    let mut stride = 1;
    while stride < HADAMARD_TRANSFORM_BLOCK_SIZE {
        for lane in 0..HADAMARD_TRANSFORM_BLOCK_SIZE {
            if lane & stride == 0 {
                let left = values[lane];
                let right = values[lane | stride];
                values[lane] = left + right;
                values[lane | stride] = left - right;
            }
        }
        stride <<= 1;
    }
    let scale = 1.0 / (HADAMARD_TRANSFORM_BLOCK_SIZE as f32).sqrt();
    for value in values {
        *value *= scale;
    }
}

#[kernel(ActivationsPrepare)]
#[variants(InputT, f32, f16, bf16)]
pub fn activations_prepare<InputT: ArrayElement + Float>(
    input: *const InputT,
    #[optional(ops.contains(ActivationPrepareOps::QUANTIZE))] q_out: Option<*mut i8>,
    #[optional(ops.contains(ActivationPrepareOps::QUANTIZE))] scales_out: Option<*mut f32>,
    #[optional(ops.contains(ActivationPrepareOps::INPUT_RHT))] rht_factors: Option<*const i32>,
    batch_size: u32,
    element_count: u32,
    group_size: u32,
    #[specialize] ops: ActivationPrepareOps,
) {
    let rows = batch_size as usize;
    let columns = element_count as usize;
    let group_size = group_size as usize;
    assert_eq!(group_size, ACTIVATION_QUANTIZATION_GROUP_SIZE as usize);
    assert!(columns.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE));
    assert_eq!(rht_factors.is_some(), ops.contains(ActivationPrepareOps::INPUT_RHT));
    assert_eq!(q_out.is_some(), ops.contains(ActivationPrepareOps::QUANTIZE));
    assert_eq!(scales_out.is_some(), ops.contains(ActivationPrepareOps::QUANTIZE));

    let (Some(q_out), Some(scales_out)) = (q_out, scales_out) else {
        return;
    };

    let groups = columns.div_ceil(group_size);
    let mut prepared = vec![0.0f32; columns];
    for row in 0..rows {
        for block_start in (0..columns).step_by(HADAMARD_TRANSFORM_BLOCK_SIZE) {
            let mut block = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE];
            for lane in 0..HADAMARD_TRANSFORM_BLOCK_SIZE {
                let index = block_start + lane;
                let value: f32 = NumCast::from(unsafe { *input.add(row * columns + index) }).unwrap();
                let factor = rht_factors.map_or(1.0, |factors| unsafe { *factors.add(index) } as f32);
                block[lane] = value * factor;
            }
            if ops.contains(ActivationPrepareOps::INPUT_RHT) {
                input_rht(&mut block);
            }
            prepared[block_start..block_start + HADAMARD_TRANSFORM_BLOCK_SIZE].copy_from_slice(&block);
        }

        for group in 0..groups {
            let start = group * group_size;
            let end = (start + group_size).min(columns);
            let slice = &prepared[start..end];
            let divisor = min_max_symmetric_divisor(slice);
            unsafe { *scales_out.add(row * groups + group) = divisor };
            for index in start..end {
                let q = quantize_symmetric_i8(prepared[index], divisor);
                unsafe { *q_out.add(row * columns + index) = q };
            }
        }
    }
}

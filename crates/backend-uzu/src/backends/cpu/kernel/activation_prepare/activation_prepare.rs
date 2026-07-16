use half::{bf16, f16};
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::common::{
        gpu_types::{ActivationPrepareOps, ActivationScaleStat},
        kernel::{group_stat, quantize_symmetric_i8, symmetric_divisor},
    },
};

const RHT_BLOCK_SIZE: usize = 32;

fn input_rht(values: &mut [f32; RHT_BLOCK_SIZE]) {
    let mut stride = 1;
    while stride < RHT_BLOCK_SIZE {
        for lane in 0..RHT_BLOCK_SIZE {
            if lane & stride == 0 {
                let left = values[lane];
                let right = values[lane | stride];
                values[lane] = left + right;
                values[lane | stride] = left - right;
            }
        }
        stride <<= 1;
    }
    let scale = 1.0 / (RHT_BLOCK_SIZE as f32).sqrt();
    for value in values {
        *value *= scale;
    }
}

#[kernel(ActivationsPrepare)]
#[variants(InputT, f32, f16, bf16)]
pub fn activations_prepare<InputT: ArrayElement + Float>(
    input: *const InputT,
    q_out: *mut i8,
    scales_out: *mut f32,
    #[optional(ops.contains(ActivationPrepareOps::INPUT_RHT))] rht_factors: Option<*const i32>,
    batch_size: u32,
    element_count: u32,
    group_size: u32,
    #[specialize] ops: ActivationPrepareOps,
    #[specialize] stat: ActivationScaleStat,
) {
    let rows = batch_size as usize;
    let columns = element_count as usize;
    let group_size = group_size as usize;
    assert!(group_size.is_multiple_of(RHT_BLOCK_SIZE));
    assert!(columns.is_multiple_of(RHT_BLOCK_SIZE));
    assert_eq!(rht_factors.is_some(), ops.contains(ActivationPrepareOps::INPUT_RHT));

    let groups = columns.div_ceil(group_size);
    let mut prepared = vec![0.0f32; columns];
    for row in 0..rows {
        for block_start in (0..columns).step_by(RHT_BLOCK_SIZE) {
            let mut block = [0.0f32; RHT_BLOCK_SIZE];
            for lane in 0..RHT_BLOCK_SIZE {
                let index = block_start + lane;
                let value: f32 = NumCast::from(unsafe { *input.add(row * columns + index) }).unwrap();
                let factor = rht_factors.map_or(1.0, |factors| unsafe { *factors.add(index) } as f32);
                block[lane] = value * factor;
            }
            if ops.contains(ActivationPrepareOps::INPUT_RHT) {
                input_rht(&mut block);
            }
            prepared[block_start..block_start + RHT_BLOCK_SIZE].copy_from_slice(&block);
        }

        for group in 0..groups {
            let start = group * group_size;
            let end = (start + group_size).min(columns);
            let divisor = symmetric_divisor(group_stat(&prepared[start..end], stat));
            unsafe { *scales_out.add(row * groups + group) = divisor };
            for index in start..end {
                unsafe { *q_out.add(row * columns + index) = quantize_symmetric_i8(prepared[index], divisor) };
            }
        }
    }
}

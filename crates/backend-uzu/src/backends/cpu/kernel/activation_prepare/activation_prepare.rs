use half::bf16;
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use super::{
    super::hadamard_transform::hadamard_transform::hadamard_transform, min_max_symmetric_divisor, quantize_symmetric_i8,
};
use crate::{array::ArrayElement, backends::common::gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE};

// Fused input RHT + groupwise symmetric int8 quantization, reference
// counterpart of the Metal `ActivationsPrepare` kernel.
#[kernel(ActivationsPrepare)]
#[variants(InputT, f32, bf16)]
pub fn activations_prepare<InputT: ArrayElement + Float>(
    input: *const InputT,
    q_out: *mut i8,
    scales_out: *mut f32,
    rht_factors: *const i32,
    batch_size: u32,
    element_count: u32,
    group_size: u32,
) {
    let rows = batch_size as usize;
    let columns = element_count as usize;
    let group_size = group_size as usize;
    assert!(columns.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE));
    assert!(group_size > 0 && columns.is_multiple_of(group_size));

    let groups = columns.div_ceil(group_size);
    let mut prepared = vec![0.0f32; columns];
    for row in 0..rows {
        for block_start in (0..columns).step_by(HADAMARD_TRANSFORM_BLOCK_SIZE) {
            let mut block = [0.0f32; HADAMARD_TRANSFORM_BLOCK_SIZE];
            for lane in 0..HADAMARD_TRANSFORM_BLOCK_SIZE {
                let index = block_start + lane;
                let value: f32 = NumCast::from(unsafe { *input.add(row * columns + index) }).unwrap();
                let factor = unsafe { *rht_factors.add(index) } as f32;
                block[lane] = value * factor;
            }
            hadamard_transform(&mut block);
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

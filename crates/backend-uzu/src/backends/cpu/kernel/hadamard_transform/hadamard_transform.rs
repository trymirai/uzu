use half::{bf16, f16};
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::{ArrayElement, backends::common::gpu_types::HadamardTransformOrder};

const SIMD_SIZE: usize = 32;

#[kernel(HadamardTransform)]
#[variants(T, f32, f16, bf16)]
pub fn hadamard_transform_mul<T: ArrayElement + Float>(
    data: *mut T,
    factors: *const i32,
    hidden_dim: u32,
    batch_size: u32,
    #[specialize] transform_order: HadamardTransformOrder,
) {
    let hidden_dim = hidden_dim as usize;
    let batch_size = batch_size as usize;
    let num_blocks = hidden_dim.div_ceil(SIMD_SIZE);
    let inv_sqrt = 1.0_f32 / (SIMD_SIZE as f32).sqrt();
    let apply_factor_first = matches!(transform_order, HadamardTransformOrder::Input);

    for batch in 0..batch_size {
        let row = batch * hidden_dim;
        for block in 0..num_blocks {
            let mut buf = [0.0_f32; SIMD_SIZE];
            for lane in 0..SIMD_SIZE {
                let col = block * SIMD_SIZE + lane;
                if col < hidden_dim {
                    let element: T = unsafe { *data.add(row + col) };
                    let element_f32: f32 = <f32 as NumCast>::from(element).unwrap_or(0.0);
                    if apply_factor_first {
                        let factor: i32 = unsafe { *factors.add(col) };
                        buf[lane] = element_f32 * (factor as f32);
                    } else {
                        buf[lane] = element_f32;
                    }
                }
            }
            for &stride in &[1usize, 2, 4, 8, 16] {
                let mut next = [0.0_f32; SIMD_SIZE];
                for lane in 0..SIMD_SIZE {
                    let partner = lane ^ stride;
                    let self_val = buf[lane];
                    let partner_val = buf[partner];
                    next[lane] = if (lane & stride) != 0 {
                        partner_val - self_val
                    } else {
                        partner_val + self_val
                    };
                }
                buf = next;
            }
            for lane in 0..SIMD_SIZE {
                let col = block * SIMD_SIZE + lane;
                if col < hidden_dim {
                    let mut value_f32 = buf[lane] * inv_sqrt;
                    if !apply_factor_first {
                        let factor: i32 = unsafe { *factors.add(col) };
                        value_f32 *= factor as f32;
                    }
                    let value: T = <T as NumCast>::from(value_f32).unwrap_or(T::zero());
                    unsafe { *data.add(row + col) = value };
                }
            }
        }
    }
}

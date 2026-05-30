use half::bf16;
use num_traits::{Float, NumCast};
use proc_macros::kernel;

use crate::{ArrayElement, backends::common::gpu_types::HadamardTransformOrder};

const SIMD_SIZE: usize = 32;

fn hadamard_transform(values: &mut [f32; SIMD_SIZE]) {
    let mut stride = 1;
    while stride < SIMD_SIZE {
        for lane in 0..SIMD_SIZE {
            if lane & stride == 0 {
                let a = values[lane];
                let b = values[lane | stride];
                values[lane] = a + b;
                values[lane | stride] = a - b;
            }
        }
        stride <<= 1;
    }
    let scale = 1.0 / (SIMD_SIZE as f32).sqrt();
    for v in values.iter_mut() {
        *v *= scale;
    }
}

#[kernel(HadamardTransform)]
#[variants(T, f32, bf16)]
pub fn hadamard_transform_mul<T: ArrayElement + Float>(
    data: *mut T,
    factors: *const i32,
    hidden_dim: u32,
    batch_size: u32,
    #[specialize] transform_order: HadamardTransformOrder,
) {
    let hidden_dim = hidden_dim as usize;
    let batch_size = batch_size as usize;
    assert_eq!(hidden_dim % SIMD_SIZE, 0, "hidden_dim must be a multiple of {SIMD_SIZE}");

    for batch in 0..batch_size {
        let row_offset = batch * hidden_dim;
        for stripe_start in (0..hidden_dim).step_by(SIMD_SIZE) {
            let mut stripe = [0.0f32; SIMD_SIZE];
            for lane in 0..SIMD_SIZE {
                let v: f32 = NumCast::from(unsafe { *data.add(row_offset + stripe_start + lane) }).unwrap();
                let f = unsafe { *factors.add(stripe_start + lane) } as f32;
                stripe[lane] = match transform_order {
                    HadamardTransformOrder::Input => v * f,
                    HadamardTransformOrder::Output => v,
                };
            }

            hadamard_transform(&mut stripe);

            for lane in 0..SIMD_SIZE {
                let f = unsafe { *factors.add(stripe_start + lane) } as f32;
                let result = match transform_order {
                    HadamardTransformOrder::Input => stripe[lane],
                    HadamardTransformOrder::Output => stripe[lane] * f,
                };
                unsafe { *data.add(row_offset + stripe_start + lane) = <T as NumCast>::from(result).unwrap() };
            }
        }
    }
}

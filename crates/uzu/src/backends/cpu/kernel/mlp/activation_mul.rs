use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{Activation, ArrayElement};

#[kernel(MlpGateActMul)]
#[variants(T, f32, f16, bf16)]
pub fn mlp_gate_act_mul<T: ArrayElement + Float>(
    fused_up: *const T,
    hidden: *mut T,
    h: i32,
    m: i32,
    act_type: crate::backends::common::gpu_types::activation_type::ActivationType,
) {
    for j in 0..h {
        for row in 0..m {
            let base = row * 2 * h;
            unsafe {
                let up: T = *fused_up.add((base + j) as usize);
                let gate: T = *fused_up.add((base + h + j) as usize);
                let g: T = act_type.activate(gate);
                let out_float = up.to_f32().unwrap() * g.to_f32().unwrap();
                *hidden.add((row * h + j) as usize) = T::from(out_float).unwrap();
            }
        }
    }
}

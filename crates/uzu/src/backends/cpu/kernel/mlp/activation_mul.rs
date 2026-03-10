use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    Activation, ArrayElement,
    backends::cpu::kernel::activation::{ActivationType, activate},
};

#[kernel(MlpGateActMul)]
#[variants(T, f32, f16, bf16)]
pub fn mlp_gate_act_mul<T: ArrayElement + Float>(
    #[allow(unused)] fused_up: *const T,
    #[allow(unused)] hidden: *mut T,
    #[allow(unused)] h: i32,
    #[allow(unused)] m: i32,
    #[allow(unused)] act_type: u32,
) {
    let activation_type = match ActivationType::try_from(act_type) {
        Ok(a) => a,
        Err(_) => return,
    };

    for j in 0..h {
        for row in 0..m {
            let base = row * 2 * h;
            unsafe {
                let up: T = *fused_up.add((base + j) as usize);
                let gate: T = *fused_up.add((base + h + j) as usize);
                let g: T = activate(gate, &activation_type);
                let out_float = up.to_f32().unwrap() * g.to_f32().unwrap();
                *hidden.add((row * h + j) as usize) = T::from(out_float).unwrap();
            }
        }
    }
}

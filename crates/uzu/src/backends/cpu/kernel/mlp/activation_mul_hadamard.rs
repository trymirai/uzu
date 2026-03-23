use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MlpGateActMulHadamard)]
#[variants(T, f32, f16, bf16)]
pub fn mlp_gate_act_mul_hadamard<T: ArrayElement + Float>(
    #[allow(unused)] fused_up: *const T,
    #[allow(unused)] hidden: *mut T,
    #[allow(unused)] hadamard_factors: *const T,
    #[allow(unused)] h: i32,
    #[allow(unused)] m: i32,
    #[allow(unused)] act_type: crate::backends::common::gpu_types::activation_type::ActivationType,
) {
    todo!()
}

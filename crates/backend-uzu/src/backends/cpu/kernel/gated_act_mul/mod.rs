use num_traits::Float;

use crate::backends::common::gpu_types::activation_type::ActivationType;

#[inline]
pub(super) fn gated_act_mul<T: Float>(
    value: T,
    gate: T,
    act_type: ActivationType,
) -> f32 {
    (value * act_type.activate(gate)).to_f32().unwrap()
}

pub mod gated_act_mul;

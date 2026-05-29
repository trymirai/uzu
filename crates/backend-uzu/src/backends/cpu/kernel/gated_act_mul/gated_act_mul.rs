use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::{ArrayElement, backends::common::gpu_types::activation_type::ActivationType};

#[kernel(GatedActMul)]
#[variants(T, f32, bf16)]
pub fn gated_act_mul<T: ArrayElement + Float>(
    act_operand: *const T,
    #[optional(!interleaved)] value_operand: Option<*const T>,
    output: *mut T,
    #[optional(use_hadamard)] hadamard_factors: Option<*const i32>,
    gated_dim: u32,
    batch_dim: u32,
    value_offset: u32,
    value_row_stride: u32,
    act_type: ActivationType,
    #[specialize] interleaved: bool,
    #[specialize] use_hadamard: bool,
) {
    if use_hadamard {
        unimplemented!("not supported yet");
    }
    let gated_dim = gated_dim as usize;
    let batch_dim = batch_dim as usize;
    let value_offset = value_offset as usize;
    let value_row_stride = value_row_stride as usize;

    for batch in 0..batch_dim {
        for gated in 0..gated_dim {
            let (act_index, value) = if interleaved {
                let base = batch * 2 * gated_dim;
                (base + gated_dim + gated, unsafe { *act_operand.add(base + gated) })
            } else {
                let value_index = batch * value_row_stride + value_offset + gated;
                (batch * gated_dim + gated, unsafe { *value_operand.unwrap().add(value_index) })
            };
            let activated: T = act_type.activate(unsafe { *act_operand.add(act_index) });
            let result = value.to_f32().unwrap() * activated.to_f32().unwrap();
            unsafe { *output.add(batch * gated_dim + gated) = T::from(result).unwrap() };
        }
    }
}

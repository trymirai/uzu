use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{ArrayElement, backends::common::gpu_types::activation_type::ActivationType};

#[kernel(GatedActMul)]
#[variants(T, f32, f16, bf16)]
pub fn gated_act_mul<T: ArrayElement + Float>(
    act_operand: *const T,
    #[optional(!interleaved)] value_operand: Option<*const T>,
    output: *mut T,
    #[optional(use_hadamard)] hadamard_factors: Option<*const i32>,
    inner_dim: i32,
    outer_dim: i32,
    value_offset: i32,
    value_row_stride: i32,
    act_type: ActivationType,
    #[specialize] interleaved: bool,
    #[specialize] use_hadamard: bool,
) {
    if use_hadamard {
        unimplemented!("not supported yet");
    }
    let inner_dim = inner_dim as usize;
    let outer_dim = outer_dim as usize;
    let value_offset = value_offset as usize;
    let value_row_stride = value_row_stride as usize;

    for outer in 0..outer_dim {
        for inner in 0..inner_dim {
            let (act_index, value) = if interleaved {
                let base = outer * 2 * inner_dim;
                (base + inner_dim + inner, unsafe { *act_operand.add(base + inner) })
            } else {
                let value_index = outer * value_row_stride + value_offset + inner;
                (outer * inner_dim + inner, unsafe { *value_operand.unwrap().add(value_index) })
            };
            let activated: T = act_type.activate(unsafe { *act_operand.add(act_index) });
            let result = value.to_f32().unwrap() * activated.to_f32().unwrap();
            unsafe { *output.add(outer * inner_dim + inner) = T::from(result).unwrap() };
        }
    }
}

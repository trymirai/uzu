use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{ArrayElement, backends::common::gpu_types::activation_type::ActivationType};

#[kernel(PleGateActMul)]
#[variants(T, f32, f16, bf16)]
pub fn ple_gate_act_mul<T: ArrayElement + Float>(
    gate_out: *const T,
    per_layer_input: *const T,
    output: *mut T,
    ple_dim: i32,
    batch_dim: i32,
    num_layers: i32,
    layer_offset: i32,
    act_type: ActivationType,
) {
    let ple_dim = ple_dim as usize;
    let batch_dim = batch_dim as usize;
    let layer_stride = num_layers as usize * ple_dim;
    let layer_offset = layer_offset as usize;

    for row in 0..batch_dim {
        for col in 0..ple_dim {
            let gate_index = row * ple_dim + col;
            let input_index = row * layer_stride + layer_offset + col;
            let gate_value = unsafe { *gate_out.add(gate_index) };
            let input_value = unsafe { *per_layer_input.add(input_index) };
            let activated: T = act_type.activate(gate_value);
            let result = activated.to_f32().unwrap() * input_value.to_f32().unwrap();
            unsafe { *output.add(gate_index) = T::from(result).unwrap() };
        }
    }
}

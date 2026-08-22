use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(SigmoidGate)]
#[variants(T, f32, f16, bf16)]
pub fn sigmoid_gate<T: ArrayElement + Float>(
    gate: *const T,
    output: *mut T,
    gate_dim: u32,
    batch_dim: u32,
    gate_row_stride: u32,
) {
    assert!(gate_dim > 0, "sigmoid gate requires nonzero gate_dim");
    assert!(gate_row_stride >= gate_dim, "sigmoid gate row stride is too small");
    let gate_dim = gate_dim as usize;
    let batch_dim = batch_dim as usize;
    let gate_row_stride = gate_row_stride as usize;
    for batch_idx in 0..batch_dim {
        for gate_idx in 0..gate_dim {
            let output_idx = batch_idx * gate_dim + gate_idx;
            let gate_idx = batch_idx * gate_row_stride + gate_idx;
            let g = unsafe { (*gate.add(gate_idx)).to_f32().unwrap() };
            let sigmoid = 1.0f32 / (1.0f32 + (-g).exp());
            let out = unsafe { (*output.add(output_idx)).to_f32().unwrap() };
            unsafe { *output.add(output_idx) = T::from(out * sigmoid).unwrap() };
        }
    }
}

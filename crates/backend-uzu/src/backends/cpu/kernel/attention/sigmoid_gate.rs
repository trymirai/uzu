use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(SigmoidGate)]
#[variants(T, f32, f16, bf16)]
pub fn sigmoid_gate<T: ArrayElement + Float>(
    gate: *const T,
    output: *mut T,
    total_elements: u32,
) {
    for idx in 0..total_elements as usize {
        unsafe {
            let g = (*gate.add(idx)).to_f32().unwrap();
            let sigmoid = 1.0f32 / (1.0f32 + (-g).exp());
            let out = (*output.add(idx)).to_f32().unwrap();
            *output.add(idx) = T::from(out * sigmoid).unwrap();
        }
    }
}

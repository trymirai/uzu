use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AttentionGate)]
#[variants(T, f32, f16, bf16)]
pub fn attention_gate<T: ArrayElement + Float>(
    qkv: *const T,
    output: *mut T,
    num_heads: u32,
    num_groups: u32,
    head_dim: u32,
    suffix_length: u32,
) {
    let qkv_stride = (2 * num_heads + 2 * num_groups) * head_dim;
    let gate_offset = (num_heads + 2 * num_groups) * head_dim;

    for token in 0..suffix_length {
        for head in 0..num_heads {
            for dim in 0..head_dim {
                let gate_idx = (token * qkv_stride + gate_offset + head * head_dim + dim) as usize;
                let output_idx = ((token * num_heads + head) * head_dim + dim) as usize;
                unsafe {
                    let g = (*qkv.add(gate_idx)).to_f32().unwrap();
                    let sigmoid = 1.0f32 / (1.0f32 + (-g).exp());
                    let out = (*output.add(output_idx)).to_f32().unwrap();
                    *output.add(output_idx) = T::from(out * sigmoid).unwrap();
                }
            }
        }
    }
}

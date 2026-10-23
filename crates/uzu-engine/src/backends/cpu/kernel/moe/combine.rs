use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

#[kernel(MoeCombine)]
#[variants(T, bf16)]
#[variants(BD, 256)]
pub fn moe_combine<T: DataTypeRepr + Float, const BD: u32>(
    expert_outputs: *const T,
    expert_weights: *const f32,
    #[optional(has_shared)] shared_output: Option<*const T>,
    #[optional(has_gate)] shared_gate: Option<*const f32>,
    output: *mut T,
    model_dim: u32,
    top_k: u32,
    batch_size: u32,
    gate_stride: u32,
    #[specialize] has_shared: bool,
    #[specialize] has_gate: bool,
) {
    let model_dim_u = model_dim as usize;
    let top_k_u = top_k as usize;
    let gate_stride_u = gate_stride as usize;
    for token_idx in 0..batch_size as usize {
        let shared_w = if has_shared {
            if has_gate {
                let g = unsafe { *shared_gate.unwrap().add(token_idx * gate_stride_u) };
                1.0f32 / (1.0f32 + (-g).exp())
            } else {
                1.0f32
            }
        } else {
            0.0f32
        };

        for col in 0..model_dim_u {
            let mut acc = 0.0f32;
            for k in 0..top_k_u {
                let w = unsafe { *expert_weights.add(token_idx * top_k_u + k) };
                let val =
                    unsafe { *expert_outputs.add((token_idx * top_k_u + k) * model_dim_u + col) }.to_f32().unwrap();
                acc += w * val;
            }
            if has_shared {
                let s_val = unsafe { *shared_output.unwrap().add(token_idx * model_dim_u + col) }.to_f32().unwrap();
                acc += shared_w * s_val;
            }
            unsafe {
                *output.add(token_idx * model_dim_u + col) = T::from(acc).unwrap();
            }
        }
    }
}

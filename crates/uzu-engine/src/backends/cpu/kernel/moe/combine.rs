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
    shared_output: *const T,
    shared_gate: *const T,
    output: *mut T,
    model_dim: u32,
    top_k: u32,
    batch_size: u32,
) {
    let model_dim = model_dim as usize;
    let top_k = top_k as usize;
    for token_idx in 0..batch_size as usize {
        let gate_logit = unsafe { *shared_gate.add(token_idx) }.to_f32().unwrap();
        let shared_weight = 1.0f32 / (1.0f32 + (-gate_logit).exp());

        for dim_idx in 0..model_dim {
            let mut acc = 0.0f32;
            for k in 0..top_k {
                let expert_weight = unsafe { *expert_weights.add(token_idx * top_k + k) };
                let expert_value =
                    unsafe { *expert_outputs.add((token_idx * top_k + k) * model_dim + dim_idx) }.to_f32().unwrap();
                acc += expert_weight * expert_value;
            }
            let shared_value = unsafe { *shared_output.add(token_idx * model_dim + dim_idx) }.to_f32().unwrap();
            acc += shared_weight * shared_value;
            unsafe {
                *output.add(token_idx * model_dim + dim_idx) = T::from(acc).unwrap();
            }
        }
    }
}

use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, backends::common::gpu_types::ActivationType};

#[kernel(MoeExpertsDecodeSinglePassB)]
#[variants(T, f32, f16, bf16)]
pub fn moe_experts_decode_single_pass_b<T: ArrayElement + Float>(
    hidden: *const f32,
    topk_ids: *const i32,
    topk_probs: *const T,
    w2_all: *const T,
    biases: *const T,
    y: *mut T,
    d_model: u32,
    d_ff: u32,
    k_input: u32,
) {
    let dm = d_model as usize;
    let df = d_ff as usize;
    let w2_expert_stride = dm * df;

    for my_col in 0..dm {
        let mut final_acc = 0.0f32;

        for k in 0..k_input as usize {
            let expert_id = unsafe { *topk_ids.add(k) };
            if expert_id < 0 {
                continue;
            }
            let expert_u = expert_id as usize;
            let prob = unsafe { *topk_probs.add(k) }.to_f32().unwrap();

            let hidden_offset = k * df;
            let w2_offset = expert_u * w2_expert_stride + my_col * df;

            let mut acc = 0.0f32;
            for h in 0..df {
                let h_val = unsafe { *hidden.add(hidden_offset + h) };
                let w_val = unsafe { *w2_all.add(w2_offset + h) }.to_f32().unwrap();
                acc += h_val * w_val;
            }

            // Add bias
            acc += unsafe { *biases.add(expert_u * dm + my_col) }.to_f32().unwrap();
            final_acc += prob * acc;
        }

        unsafe {
            *y.add(my_col) = T::from(final_acc).unwrap();
        }
    }
}

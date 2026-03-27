use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, backends::common::gpu_types::ActivationType};

#[kernel(MoeExpertsDecodeSinglePassA)]
#[variants(T, f32, f16, bf16)]
pub fn moe_experts_decode_single_pass_a<T: ArrayElement + Float>(
    x: *const T,
    topk_ids: *const i32,
    w13_all: *const T,
    biases: *const T,
    hidden_out: *mut f32,
    d_model: u32,
    d_ff: u32,
    k: u32,
    silu_alpha: f32,
    gate_clip_min: f32,
    gate_clip_max: f32,
    up_clip_min: f32,
    up_clip_max: f32,
    #[specialize] gating_sel: u32,
) {
    let dm = d_model as usize;
    let df = d_ff as usize;

    for k_slot in 0..k as usize {
        let expert_id = unsafe { *topk_ids.add(k_slot) };
        if expert_id < 0 {
            continue;
        }
        let expert_u = expert_id as usize;

        let w13_stride = dm * 2 * df;
        let w13_base = expert_u * w13_stride;
        let bias_base = expert_u * 2 * df;

        for h_idx in 0..df {
            // Up projection: dot(x, w_up_row) + bias
            let w_up_offset = w13_base + h_idx * dm;
            let mut acc_up = 0.0f32;
            for d in 0..dm {
                let x_val = unsafe { *x.add(d) }.to_f32().unwrap();
                let w_val = unsafe { *w13_all.add(w_up_offset + d) }.to_f32().unwrap();
                acc_up += x_val * w_val;
            }

            // Gate projection (only for SwiGLU/GEGLU)
            let mut acc_gate = 0.0f32;
            if gating_sel > 1 {
                let w_gate_offset = w13_base + (df + h_idx) * dm;
                for d in 0..dm {
                    let x_val = unsafe { *x.add(d) }.to_f32().unwrap();
                    let w_val = unsafe { *w13_all.add(w_gate_offset + d) }.to_f32().unwrap();
                    acc_gate += x_val * w_val;
                }
            }

            // Add bias and clip, then activate
            let activated = if gating_sel <= 1 {
                let up_val = (acc_up + unsafe { *biases.add(bias_base + h_idx) }.to_f32().unwrap())
                    .clamp(up_clip_min, up_clip_max);
                if gating_sel == 0 {
                    ActivationType::GELU.activate(up_val)
                } else {
                    ActivationType::SILU.activate(silu_alpha * up_val)
                }
            } else {
                let up_val = (acc_up + unsafe { *biases.add(bias_base + h_idx) }.to_f32().unwrap())
                    .clamp(up_clip_min, up_clip_max);
                let gate_val = (acc_gate + unsafe { *biases.add(bias_base + df + h_idx) }.to_f32().unwrap())
                    .clamp(gate_clip_min, gate_clip_max);
                let gate_act = if gating_sel == 2 {
                    ActivationType::SILU.activate(gate_val * silu_alpha)
                } else {
                    ActivationType::GELU.activate(gate_val)
                };
                gate_act * up_val
            };

            unsafe {
                *hidden_out.add(k_slot * df + h_idx) = activated;
            }
        }
    }
}

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

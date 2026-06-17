use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::array::ArrayElement;

#[kernel(MoeRouterTopK)]
#[variants(ScalarT, f16, bf16, f32)]
pub fn moe_router_top_k<ScalarT: ArrayElement + Float>(
    input: *const ScalarT,
    weight: *const ScalarT,
    bias: *const ScalarT,
    router_scale: *const ScalarT,
    per_expert_scale: *const ScalarT,
    topk_ids: *mut i32,
    topk_probs: *mut ScalarT,
    t: u32,
    d_model: u32,
    e: u32,
    k: u32,
    renorm: bool,
    router_norm_epsilon: f32,
    router_input_scale: f32,
    has_biases: bool,
    has_router_scales: bool,
    has_per_expert_scales: bool,
    normalize_router_input: bool,
) {
    assert_eq!(d_model % 4, 0, "d_model must be multiple of 4");
    assert!(k >= 1 && e >= k);

    let t = t as usize;
    let d_model = d_model as usize;
    let e = e as usize;
    let k = k as usize;

    let mut logits = vec![0.0f32; t * e];
    unsafe {
        for token in 0..t {
            let x_row = input.add(token * d_model);
            let mut inv_rms = 1.0f32;
            if normalize_router_input {
                let mut sum_sq = 0.0f32;
                for d in 0..d_model {
                    let x = (*x_row.add(d)).to_f32().unwrap();
                    sum_sq += x * x;
                }
                inv_rms = (sum_sq / d_model as f32 + router_norm_epsilon).sqrt().recip();
            }
            for expert in 0..e {
                let w_row = weight.add(expert * d_model);
                let mut accum = [0.0f32; 4];

                // Simulate GPU vec4 processing: accumulate in 4-element chunks
                for chunk in (0..d_model).step_by(4) {
                    for i in 0..4 {
                        let d = chunk + i;
                        let scale = if has_router_scales {
                            (*router_scale.add(d)).to_f32().unwrap()
                        } else {
                            1.0
                        };
                        let x = (*x_row.add(d)).to_f32().unwrap() * inv_rms * router_input_scale * scale;
                        accum[i] += (*w_row.add(d)).to_f32().unwrap() * x;
                    }
                }

                // Sum the 4-vector: (a.x + a.y) + (a.z + a.w) - matches Metal shader line 60
                let sum = (accum[0] + accum[1]) + (accum[2] + accum[3]);
                logits[token * e + expert] = sum
                    + if has_biases {
                        (*bias.add(expert)).to_f32().unwrap()
                    } else {
                        0.0
                    };
            }

            let mut best_vals = vec![f32::NEG_INFINITY; k];
            let mut best_ids = vec![-1i32; k];
            let row = &logits[token * e..(token + 1) * e];
            for expert in 0..e {
                let v = row[expert];
                let mut insert_pos = None;
                for j in (0..k).rev() {
                    if v > best_vals[j] || (v == best_vals[j] && (best_ids[j] < 0 || (expert as i32) < best_ids[j])) {
                        insert_pos = Some(j);
                    }
                }
                if let Some(pos) = insert_pos {
                    for s in (pos + 1..k).rev() {
                        best_vals[s] = best_vals[s - 1];
                        best_ids[s] = best_ids[s - 1];
                    }
                    best_vals[pos] = v;
                    best_ids[pos] = expert as i32;
                }
            }
            let base = token * k;
            for kk in 0..k {
                *topk_ids.add(base + kk) = best_ids[kk];
            }
            if renorm {
                let max_v = best_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exps = vec![0.0f32; k];
                let mut sum = 0.0f32;
                for kk in 0..k {
                    exps[kk] = (best_vals[kk] - max_v).exp();
                    sum += exps[kk];
                }
                if sum > 0.0 {
                    for kk in 0..k {
                        let expert_scale = if has_per_expert_scales {
                            (*per_expert_scale.add(best_ids[kk] as usize)).to_f32().unwrap()
                        } else {
                            1.0
                        };
                        *topk_probs.add(base + kk) = ScalarT::from(exps[kk] / sum * expert_scale).unwrap();
                    }
                } else {
                    let uniform = 1.0f32 / k as f32;
                    for kk in 0..k {
                        let expert_scale = if has_per_expert_scales {
                            (*per_expert_scale.add(best_ids[kk] as usize)).to_f32().unwrap()
                        } else {
                            1.0
                        };
                        *topk_probs.add(base + kk) = ScalarT::from(uniform * expert_scale).unwrap();
                    }
                }
            } else {
                for kk in 0..k {
                    let expert_scale = if has_per_expert_scales {
                        (*per_expert_scale.add(best_ids[kk] as usize)).to_f32().unwrap()
                    } else {
                        1.0
                    };
                    *topk_probs.add(base + kk) = ScalarT::from(best_vals[kk] * expert_scale).unwrap();
                }
            }
        }
    }
}

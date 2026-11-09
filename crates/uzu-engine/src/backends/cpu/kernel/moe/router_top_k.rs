use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{backends::common::gpu_types::moe, data_type::DataTypeRepr};

#[kernel(MoeRouterTopK)]
#[variants(T, f32)]
#[variants(MAX_TOP_K, 8)]
pub fn moe_router_top_k<T: DataTypeRepr + Float, const MAX_TOP_K: u32>(
    logits: *const T,
    expert_ids: *mut u32,
    expert_weights: *mut f32,
    num_experts: u32,
    top_k: u32,
    batch_size: u32,
) {
    assert!(num_experts <= moe::ROUTER_MAX_EXPERTS);
    assert!(top_k > 0 && top_k <= MAX_TOP_K && top_k <= num_experts);
    let num_experts = num_experts as usize;
    let top_k = top_k as usize;
    for token_idx in 0..batch_size as usize {
        let mut pairs: Vec<(f32, u32)> = (0..num_experts)
            .map(|expert_idx| {
                let mut logit = unsafe { *logits.add(token_idx * num_experts + expert_idx) }.to_f32().unwrap();
                if logit.is_nan() {
                    logit = f32::NEG_INFINITY;
                }
                (logit, expert_idx as u32)
            })
            .collect();

        pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let chosen = &pairs[..top_k];
        let max_chosen = chosen[0].0; // the picks are descending
        let mut exps = Vec::with_capacity(top_k);
        let mut sum_exp = 0.0f32;
        for &(logit, _) in chosen {
            let weight = if max_chosen == f32::NEG_INFINITY {
                1.0f32 / (top_k as f32)
            } else {
                (logit - max_chosen).exp()
            };
            exps.push(weight);
            sum_exp += weight;
        }
        let inv_sum = if sum_exp > 0.0 {
            1.0f32 / sum_exp
        } else {
            0.0
        };
        for k in 0..top_k {
            unsafe {
                *expert_ids.add(token_idx * top_k + k) = chosen[k].1;
                *expert_weights.add(token_idx * top_k + k) = exps[k] * inv_sum;
            }
        }
    }
}

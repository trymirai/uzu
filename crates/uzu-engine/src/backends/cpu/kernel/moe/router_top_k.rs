use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

#[kernel(MoeRouterTopK)]
#[variants(T, f32)]
#[variants(MAX_TOP_K, 8)]
pub fn moe_router_top_k<T: DataTypeRepr + Float, const MAX_TOP_K: u32>(
    logits: *const T,
    #[optional(has_biases)] biases: Option<*const T>,
    expert_ids: *mut u32,
    expert_weights: *mut f32,
    num_experts: u32,
    top_k: u32,
    batch_size: u32,
    #[specialize] has_biases: bool,
) {
    let num_experts_usize = num_experts as usize;
    let top_k_usize = top_k as usize;
    for token_idx in 0..batch_size as usize {
        let mut pairs: Vec<(f32, u32)> = (0..num_experts_usize)
            .map(|e| {
                let mut v = unsafe { *logits.add(token_idx * num_experts_usize + e) }.to_f32().unwrap();
                if has_biases {
                    v += unsafe { *biases.unwrap().add(e) }.to_f32().unwrap();
                }
                if v.is_nan() {
                    v = f32::NEG_INFINITY;
                }
                (v, e as u32)
            })
            .collect();

        pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let chosen = &pairs[..top_k_usize];
        let max_val = chosen.iter().map(|&(v, _)| v).fold(f32::NEG_INFINITY, f32::max);
        let mut exps = Vec::with_capacity(top_k_usize);
        let mut sum_exp = 0.0f32;
        for &(v, _) in chosen {
            let ev = if max_val == f32::NEG_INFINITY {
                1.0f32 / (top_k_usize as f32)
            } else {
                (v - max_val).exp()
            };
            exps.push(ev);
            sum_exp += ev;
        }
        let inv_sum = if sum_exp > 0.0 {
            1.0f32 / sum_exp
        } else {
            0.0
        };
        for k in 0..top_k_usize {
            unsafe {
                *expert_ids.add(token_idx * top_k_usize + k) = chosen[k].1;
                *expert_weights.add(token_idx * top_k_usize + k) = exps[k] * inv_sum;
            }
        }
    }
}

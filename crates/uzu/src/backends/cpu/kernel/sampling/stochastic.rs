use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, language_model::gumbel};

// CPU stub for Stochastic — delegates to exact per-token Gumbel-max sampling.
// Full CPU optimisation deferred; this ensures trait completeness for benchmarking the Metal path.
#[kernel(Stochastic)]
#[variants(T, f32, f16, bf16)]
pub fn stochastic<T: ArrayElement + Float>(
    logits: *const T,
    batch_seeds: *const u64,
    sampled_tokens: *mut u32,
    #[optional(has_bitmask)] bitmask: Option<*const u32>,
    batch_size: u32,
    vocab_size: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    #[specialize] has_bitmask: bool,
) {
    let _ = min_p;
    let vocab_size = vocab_size as usize;
    let top_k = if top_k == 0 {
        vocab_size
    } else {
        top_k as usize
    };
    let top_p = if top_p <= 0.0 {
        1.0f32
    } else {
        top_p
    };
    let bitmask_stride = (vocab_size + 31) / 32;

    let masked = |batch_idx: usize, i: usize| -> f32 {
        let v = unsafe { (*logits.add(batch_idx * vocab_size + i)).to_f32().unwrap() / temperature };
        if has_bitmask {
            let bitmask = bitmask.unwrap();
            let word = unsafe { *bitmask.add(batch_idx * bitmask_stride + i / 32) };
            if (word >> (i % 32)) & 1 == 0 {
                return f32::NEG_INFINITY;
            }
        }
        v
    };

    for batch_idx in 0..batch_size as usize {
        let seed = unsafe { *batch_seeds.add(batch_idx) };

        // ── Phase 0: global max for numerics
        let max_logit =
            (0..vocab_size).map(|i| masked(batch_idx, i)).filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);

        if max_logit == f32::NEG_INFINITY {
            unsafe { *sampled_tokens.add(batch_idx) = 0 };
            continue;
        }

        // ── Phase 1: compute total sum_exp, top_k/top_p threshold
        let mut pairs: Vec<(f32, usize)> = (0..vocab_size)
            .filter_map(|i| {
                let v = masked(batch_idx, i);
                if v.is_finite() {
                    Some((v, i))
                } else {
                    None
                }
            })
            .collect();

        pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let total_sum_exp: f32 = pairs.iter().map(|(v, _)| (v - max_logit).exp()).sum();
        let top_p_mass = top_p * total_sum_exp;

        let mut cumulative_exp = 0.0f32;
        let mut threshold = f32::NEG_INFINITY;
        for (k, (v, _)) in pairs.iter().enumerate() {
            cumulative_exp += (v - max_logit).exp();
            if k + 1 >= top_k || cumulative_exp >= top_p_mass {
                threshold = *v;
                break;
            }
        }

        // ── Phase 2: Gumbel-max argmax over tokens above threshold
        let best_idx = (0..vocab_size)
            .filter_map(|i| {
                let v = masked(batch_idx, i);
                if v.is_finite() && v >= threshold {
                    let (offset, word) = gumbel::revidx(i as u32);
                    let noisy = v + gumbel::gumbel_float(seed, (offset, word));
                    Some((noisy, i))
                } else {
                    None
                }
            })
            .fold((f32::NEG_INFINITY, 0usize), |acc, x| {
                if x.0 > acc.0 {
                    x
                } else {
                    acc
                }
            })
            .1;

        unsafe { *sampled_tokens.add(batch_idx) = best_idx as u32 };
    }
}

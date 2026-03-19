use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, language_model::gumbel::uniform_float};

// CPU implementation of FlashInfer-style dual-pivot rejection sampling
// with logit-space pivots. Mirrors the Metal kernel logic.
#[kernel(UnifiedStochastic)]
#[variants(T, f32, f16, bf16)]
pub fn unified_stochastic<T: ArrayElement + Float>(
    logits: *const T,
    batch_seeds: *const u64,
    sampled_tokens: *mut u32,
    batch_size: u32,
    vocab_size: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
) {
    let vocab_size = vocab_size as usize;

    for batch_idx in 0..batch_size as usize {
        let batch_start = batch_idx * vocab_size;
        let seed = unsafe { *batch_seeds.add(batch_idx) };

        // ── Phase 0: max logit ────────────────────────────────────────────────
        let max_logit = (0..vocab_size)
            .map(|i| unsafe { (*logits.add(batch_start + i)).to_f32().unwrap() / temperature })
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);

        if max_logit == f32::NEG_INFINITY {
            unsafe { *sampled_tokens.add(batch_idx) = 0 };
            continue;
        }

        // ── Phase 1: sum_exp over min_p-filtered tokens ───────────────────────
        let min_p_thresh = if min_p > 0.0 { max_logit + min_p.ln() } else { f32::NEG_INFINITY };

        let sum_exp: f32 = (0..vocab_size)
            .map(|i| unsafe { (*logits.add(batch_start + i)).to_f32().unwrap() / temperature })
            .filter(|v| v.is_finite() && *v >= min_p_thresh)
            .map(|v| (v - max_logit).exp())
            .sum();

        if sum_exp <= 0.0 {
            unsafe { *sampled_tokens.add(batch_idx) = 0 };
            continue;
        }

        let top_p_mass = top_p * sum_exp;

        // ── FlashInfer dual-pivot rejection loop (logit space) ────────────────
        let mut low_logit: f32 = min_p_thresh;
        let mut high_logit: f32 = max_logit;
        let mut q_unnorm: f32 = sum_exp;
        let mut sampled_id: usize = 0;

        for round in 0..32u32 {
            let u = f32::max(uniform_float(seed, (round, 0)), 1e-37) * q_unnorm;

            // ── A: Inverse-transform sample ───────────────────────────────────
            // exp() only called for tokens with logit >= low_logit
            let mut cum = 0.0f32;
            let mut found_id = 0usize;
            let mut found_v = low_logit;
            let mut got_any = false;

            for i in 0..vocab_size {
                let v = unsafe { (*logits.add(batch_start + i)).to_f32().unwrap() } / temperature;
                if v.is_finite() && v >= low_logit {
                    let unnorm = (v - max_logit).exp();
                    if !got_any {
                        found_id = i;
                        found_v = v;
                        got_any = true;
                    }
                    cum += unnorm;
                    if cum > u {
                        found_id = i;
                        found_v = v;
                        break;
                    }
                }
            }
            sampled_id = found_id;

            if top_k == 0 && top_p >= 1.0 {
                break;
            }

            // ── B: Aggregate unnorm mass and count above each logit pivot ─────
            let pivot0 = found_v;
            let pivot1 = (pivot0 + high_logit) * 0.5;

            let mut agg0 = 0.0f32;
            let mut cnt0 = 0.0f32;
            let mut agg1 = 0.0f32;
            let mut cnt1 = 0.0f32;

            for i in 0..vocab_size {
                let v = unsafe { (*logits.add(batch_start + i)).to_f32().unwrap() } / temperature;
                if v.is_finite() && v >= low_logit {
                    let unnorm = (v - max_logit).exp();
                    if v > pivot0 { agg0 += unnorm; cnt0 += 1.0; }
                    if v > pivot1 { agg1 += unnorm; cnt1 += 1.0; }
                }
            }

            // ── C: Accept or narrow ───────────────────────────────────────────
            let ok_k0 = top_k == 0 || cnt0 < top_k as f32;
            let ok_p0 = top_p >= 1.0 || agg0 < top_p_mass;
            let ok_k1 = top_k == 0 || cnt1 < top_k as f32;
            let ok_p1 = top_p >= 1.0 || agg1 < top_p_mass;

            if ok_k0 && ok_p0 {
                break;
            } else if ok_k1 && ok_p1 {
                low_logit = pivot0;
                high_logit = pivot1;
                q_unnorm = agg0;
            } else {
                low_logit = pivot1;
                q_unnorm = agg1;
            }
        }

        unsafe { *sampled_tokens.add(batch_idx) = sampled_id as u32 };
    }
}

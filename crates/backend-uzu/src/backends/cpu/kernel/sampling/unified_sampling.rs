use std::cmp::Ordering;

use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::{
    array::ArrayElement,
    encodable_block::sampling::{gumbel_float, revidx},
};

const CANDIDATE_SEED: usize = 64;
const CANDIDATE_GROWTH: usize = 8;

// NOTE: top_k + top_p combination is not exactly matching lalamo ("parallel" here, should be top-k then top-p)
#[kernel(UnifiedSampling)]
#[variants(T, f32, bf16)]
pub fn unified_sampling<T: ArrayElement + Float>(
    logits: *const T,
    output: *mut u32,
    #[optional(is_stochastic)] seeds: Option<*const u64>,
    #[optional(has_bitmask)] bitmask: Option<*const u32>,
    #[optional(has_temperature)] temperature: Option<f32>,
    #[optional(has_top_k)] top_k: Option<u32>,
    #[optional(has_top_p)] top_p: Option<f32>,
    #[optional(has_min_p)] min_p: Option<f32>,
    vocab_size: u32,
    batch_size: u32,
    #[specialize] is_stochastic: bool,
    #[specialize] has_bitmask: bool,
    #[specialize] has_temperature: bool,
    #[specialize] has_top_k: bool,
    #[specialize] has_top_p: bool,
    #[specialize] has_min_p: bool,
) {
    let vocab = vocab_size as usize;
    debug_assert!(vocab > 0, "vocab_size must be positive");
    let filtered = has_top_k || has_top_p || has_min_p;

    let mut scores = vec![0.0f32; vocab];
    let mut candidates: Vec<u32> = if filtered {
        (0..vocab_size).collect()
    } else {
        Vec::new()
    };
    let mut kept: Vec<(u32, f32)> = Vec::new();

    for batch_idx in 0..batch_size {
        let row = unsafe { std::slice::from_raw_parts(logits.wrapping_add((vocab_size * batch_idx) as usize), vocab) };
        for (score, logit) in scores.iter_mut().zip(row) {
            *score = logit.to_f32().unwrap();
        }

        if has_bitmask {
            let bitmask = unsafe {
                std::slice::from_raw_parts(
                    bitmask.unwrap().wrapping_add((vocab_size.div_ceil(u32::BITS) * batch_idx) as usize),
                    vocab_size.div_ceil(u32::BITS) as usize,
                )
            };
            for (logit_index, logit) in scores.iter_mut().enumerate() {
                if bitmask[logit_index / (u32::BITS as usize)] & (1 << (logit_index % (u32::BITS as usize))) == 0 {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }

        if has_temperature {
            let recip_temperature = 1.0 / temperature.unwrap();
            for logit in scores.iter_mut() {
                *logit *= recip_temperature;
            }
        }

        if filtered {
            filter_candidates(
                &mut scores,
                &mut candidates,
                &mut kept,
                has_top_k.then(|| top_k.unwrap()),
                has_top_p.then(|| top_p.unwrap()),
                has_min_p.then(|| min_p.unwrap()),
            );
        }

        if is_stochastic {
            let seed = unsafe { *seeds.unwrap().wrapping_add(batch_idx as usize) };
            for (logit_index, logit) in scores.iter_mut().enumerate() {
                if *logit != f32::NEG_INFINITY {
                    *logit += gumbel_float(seed, revidx(logit_index as u32, vocab_size));
                }
            }
        }

        let mut argmax = 0usize;
        let mut best = scores[0];
        for (index, &value) in scores.iter().enumerate().skip(1) {
            if value > best {
                best = value;
                argmax = index;
            }
        }

        unsafe { *output.wrapping_add(batch_idx as usize) = argmax as u32 }
    }
}

#[inline(always)]
fn candidate_cmp(
    scores: &[f32],
    left: u32,
    right: u32,
) -> Ordering {
    scores[right as usize].partial_cmp(&scores[left as usize]).unwrap_or(Ordering::Equal).then(left.cmp(&right))
}

fn filter_candidates(
    scores: &mut [f32],
    candidates: &mut [u32],
    kept: &mut Vec<(u32, f32)>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
) {
    let vocab = scores.len();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut limit = vocab;
    if let Some(top_k) = top_k {
        limit = limit.min(top_k as usize);
    }
    let min_p_threshold = min_p.map(|min_p| max + min_p.ln());
    if let Some(threshold) = min_p_threshold
        && threshold.is_finite()
    {
        limit = limit.min(scores.iter().filter(|score| **score >= threshold).count().max(1));
    }
    limit = limit.max(1);

    let norm = top_p.map(|_| {
        let mut sum = 0.0f32;
        let mut compensation = 0.0f32;
        for &score in scores.iter() {
            let term = (score - max).exp() - compensation;
            let next = sum + term;
            compensation = (next - sum) - term;
            sum = next;
        }
        sum
    });

    let mut cap = if top_p.is_some() {
        CANDIDATE_SEED.min(limit)
    } else {
        limit
    };

    loop {
        if cap < vocab {
            candidates.select_nth_unstable_by(cap - 1, |left, right| candidate_cmp(scores, *left, *right));
        }
        candidates[..cap].sort_unstable_by(|left, right| candidate_cmp(scores, *left, *right));

        kept.clear();
        let mut mass = 0.0f32;
        let mut stopped = false;
        for (rank, &index) in candidates[..cap].iter().enumerate() {
            let value = scores[index as usize];
            if top_k.is_some_and(|top_k| rank as u32 >= top_k)
                || top_p.is_some_and(|top_p| mass >= top_p)
                || min_p_threshold.is_some_and(|threshold| value < threshold)
            {
                stopped = true;
                break;
            }
            kept.push((index, value));
            if let Some(norm) = norm {
                mass += (value - max).exp() / norm;
            }
        }

        if stopped || cap == limit {
            break;
        }
        cap = (cap * CANDIDATE_GROWTH).min(limit);
    }

    scores.fill(f32::NEG_INFINITY);
    for &(index, value) in kept.iter() {
        scores[index as usize] = value;
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/backends/cpu/kernel/sampling/unified_sampling_test.rs"]
mod tests;

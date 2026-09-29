use half::bf16;
use uzu_engine_macros::kernel;

use crate::{
    backends::common::gpu_types::weaver::{CANDIDATES_MAX, DraftSamplingParams, FRONTIER_NO_WINNER, MetadataIdx},
    encodable_block::sampling::{gumbel_float, revidx},
};

// The target sampler's filters over the pool, as the sorted prefix of unified_sampling.rs, with ties broken by
// token id (-0 and +0 are equal) instead of the vocabulary index.
fn draft_sampling_mask(
    logits: &[f32],
    token: impl Fn(usize) -> u32,
    params: &DraftSamplingParams,
) -> Vec<bool> {
    let canonical = |value: f32| {
        if value == 0.0 {
            0.0
        } else {
            value
        }
    };
    let mut order = (0..logits.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        canonical(logits[right]).total_cmp(&canonical(logits[left])).then_with(|| token(left).cmp(&token(right)))
    });
    let logits_max = logits[order[0]];
    let logits_norm = order.iter().map(|&index| (logits[index] - logits_max).exp()).sum::<f32>();
    let mut keep = vec![false; logits.len()];
    let mut top_p_mass = 0.0;
    for (top_k_num, &index) in order.iter().enumerate() {
        if top_k_num as u32 >= params.top_k
            || top_p_mass >= params.top_p
            || (params.min_p > 0.0 && logits[index] < logits_max + params.min_p.ln())
        {
            break;
        }
        keep[index] = true;
        top_p_mass += (logits[index] - logits_max).exp() / logits_norm;
    }
    keep
}

#[kernel(WeaverTopChildren)]
pub fn weaver_top_children(
    residual_logits: *const bf16,
    candidate_logits: *const f32,
    candidate_ids: *const u32,
    depth_seeds: *const u64,
    node_metadata: *const u32,
    output_token_ids: *mut u32,
    output_model_logprobs: *mut f32,
    #[optional(has_prune_noise)] output_prune_logprobs: Option<*mut f32>,
    rows: u32,
    candidates: u32,
    expand_width: u32,
    vocab_size: u32,
    #[optional(has_prune_noise)] prune_noise_scale: Option<f32>,
    #[optional(has_draft_sampling)] draft_sampling_params: Option<
        crate::backends::common::gpu_types::weaver::DraftSamplingParams,
    >,
    #[specialize] has_prune_noise: bool,
    #[specialize] has_draft_sampling: bool,
) {
    let rows = rows as usize;
    let candidates = candidates as usize;
    let expand_width = expand_width as usize;
    if candidates == 0 || candidates > CANDIDATES_MAX as usize || expand_width == 0 || expand_width > candidates {
        return;
    }
    for row in 0..rows {
        let base = row * candidates;
        let depth = unsafe { *node_metadata.add(MetadataIdx::Depth as usize * rows + row) } as usize;
        let seed = unsafe { *depth_seeds.add(depth) };
        let token = |index: usize| unsafe { *candidate_ids.add(base + index) };
        // Draft sampling follows the target's temperature in selection, expansion and pruning alike.
        let draft_sampling = has_draft_sampling.then(|| draft_sampling_params.unwrap());
        let logits = (0..candidates)
            .map(|index| {
                let logit =
                    unsafe { *candidate_logits.add(base + index) + (*residual_logits.add(base + index)).to_f32() };
                draft_sampling.map_or(logit, |params| logit * params.recip_temperature)
            })
            .collect::<Vec<_>>();
        let live = draft_sampling
            .map_or_else(|| vec![true; candidates], |params| draft_sampling_mask(&logits, token, &params));
        let live_logits = || logits.iter().zip(&live).filter(|(_, live)| **live).map(|(logit, _)| *logit);
        let max = live_logits().fold(f32::NEG_INFINITY, f32::max);
        let log_sum = live_logits().map(|value| (value - max).exp()).sum::<f32>().ln() + max;
        let perturbed = (0..candidates)
            .map(|index| logits[index] + gumbel_float(seed, revidx(token(index), vocab_size)))
            .collect::<Vec<_>>();
        // Final pruning weighs each edge by the pool softmax of logits plus the target's noise scaled by 1 / sigma.
        let prune = has_prune_noise.then(|| {
            let scale = prune_noise_scale.unwrap();
            let prune_logits = (0..candidates)
                .map(|index| logits[index] + scale * gumbel_float(seed, revidx(token(index), vocab_size)))
                .collect::<Vec<_>>();
            let live_prune_logits =
                || prune_logits.iter().zip(&live).filter(|(_, live)| **live).map(|(logit, _)| *logit);
            let prune_max = live_prune_logits().fold(f32::NEG_INFINITY, f32::max);
            let prune_log_sum =
                live_prune_logits().map(|value| (value - prune_max).exp()).sum::<f32>().ln() + prune_max;
            (prune_logits, prune_log_sum)
        });
        let mut indices = (0..candidates).filter(|&index| live[index]).collect::<Vec<_>>();
        indices.sort_by(|&left, &right| {
            perturbed[right].total_cmp(&perturbed[left]).then_with(|| token(left).cmp(&token(right)))
        });
        for rank in 0..expand_width {
            let output = row * expand_width + rank;
            let Some(&index) = indices.get(rank) else {
                // Fewer candidates survived the filters than there are children; the sentinel keeps the slot
                // out of the frontier.
                unsafe {
                    *output_token_ids.add(output) = FRONTIER_NO_WINNER;
                    *output_model_logprobs.add(output) = f32::NEG_INFINITY;
                    if let Some(output_prune_logprobs) = output_prune_logprobs {
                        *output_prune_logprobs.add(output) = f32::NEG_INFINITY;
                    }
                }
                continue;
            };
            unsafe {
                *output_token_ids.add(output) = token(index);
                *output_model_logprobs.add(output) = logits[index] - log_sum;
                if let Some((prune_logits, prune_log_sum)) = &prune {
                    *output_prune_logprobs.unwrap().add(output) = prune_logits[index] - prune_log_sum;
                }
            }
        }
    }
}

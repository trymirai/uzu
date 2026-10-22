use half::bf16;
use uzu_engine_macros::kernel;

use crate::{
    backends::common::gpu_types::weaver::{CANDIDATES_MAX, FrontierIdx, MetadataIdx, TreeIdx},
    encodable_block::sampling::{gumbel_float, revidx},
};

#[kernel(WeaverTopChildren)]
pub fn weaver_top_children(
    residual_logits: *const bf16,
    candidate_logits: *const f32,
    candidate_ids: *const u32,
    depth_seeds: *const u64,
    node_metadata: *const u32,
    packed_tree: *const u32,
    node_valid: *const u32,
    frontier: *mut u32,
    rows: u32,
    candidates: u32,
    expand_width: u32,
    vocab_size: u32,
    frontier_capacity: u32,
    tree_slot_count: u32,
    #[optional(has_prune_noise)] prune_noise_scale: Option<f32>,
    #[specialize] has_prune_noise: bool,
) {
    let rows = rows as usize;
    let candidates = candidates as usize;
    let expand_width = expand_width as usize;
    let frontier_capacity = frontier_capacity as usize;
    let tree_slot_count = tree_slot_count as usize;
    if candidates == 0
        || candidates > CANDIDATES_MAX as usize
        || expand_width == 0
        || expand_width > candidates
        || frontier_capacity == 0
        || tree_slot_count == 0
    {
        return;
    }
    for row in 0..rows {
        if unsafe { *node_valid.add(row) == 0 } {
            continue;
        }
        let tree_slot = unsafe { *node_metadata.add(MetadataIdx::TreeSlot as usize * rows + row) as usize };
        if tree_slot >= tree_slot_count {
            continue;
        }
        let base = row * candidates;
        let depth = unsafe { *node_metadata.add(MetadataIdx::Depth as usize * rows + row) } as usize;
        let seed = unsafe { *depth_seeds.add(depth) };
        let token = |index: usize| unsafe { *candidate_ids.add(base + index) };
        let logits = (0..candidates)
            .map(|index| unsafe { *candidate_logits.add(base + index) + (*residual_logits.add(base + index)).to_f32() })
            .collect::<Vec<_>>();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum = logits.iter().map(|value| (value - max).exp()).sum::<f32>().ln() + max;
        let perturbed = (0..candidates)
            .map(|index| logits[index] + gumbel_float(seed, revidx(token(index), vocab_size)))
            .collect::<Vec<_>>();
        // Final pruning weighs each edge by the pool softmax of logits plus the target's noise scaled by 1 / sigma.
        let prune = has_prune_noise.then(|| {
            let scale = prune_noise_scale.unwrap();
            let prune_logits = (0..candidates)
                .map(|index| logits[index] + scale * gumbel_float(seed, revidx(token(index), vocab_size)))
                .collect::<Vec<_>>();
            let prune_max = prune_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let prune_log_sum =
                prune_logits.iter().map(|value| (value - prune_max).exp()).sum::<f32>().ln() + prune_max;
            (prune_logits, prune_log_sum)
        });
        let mut indices = (0..candidates).collect::<Vec<_>>();
        indices.sort_by(|&left, &right| {
            perturbed[right].total_cmp(&perturbed[left]).then_with(|| token(left).cmp(&token(right)))
        });
        for (rank, index) in indices.into_iter().take(expand_width).enumerate() {
            let slot = tree_slot * expand_width + rank;
            if slot >= frontier_capacity {
                continue;
            }
            let model_logprob = logits[index] - log_sum;
            unsafe {
                let path_logprob =
                    f32::from_bits(*packed_tree.add(TreeIdx::PathLogprobBits as usize * tree_slot_count + tree_slot));
                let depth = *packed_tree.add(TreeIdx::Depth as usize * tree_slot_count + tree_slot) + 1;
                let cumulative_logprob = path_logprob + model_logprob;
                let edge_logprob = prune
                    .as_ref()
                    .map_or(model_logprob, |(prune_logits, prune_log_sum)| prune_logits[index] - prune_log_sum);
                *frontier.add(FrontierIdx::TokenId as usize * frontier_capacity + slot) = token(index);
                *frontier.add(FrontierIdx::ParentSlot as usize * frontier_capacity + slot) = tree_slot as u32;
                *frontier.add(FrontierIdx::Depth as usize * frontier_capacity + slot) = depth;
                *frontier.add(FrontierIdx::PathLogprobBits as usize * frontier_capacity + slot) =
                    cumulative_logprob.to_bits();
                *frontier.add(FrontierIdx::EdgeLogprobBits as usize * frontier_capacity + slot) =
                    edge_logprob.to_bits();
                *frontier.add(FrontierIdx::PathScoreKey as usize * frontier_capacity + slot) =
                    top_k_score_key(cumulative_logprob);
                *frontier.add(FrontierIdx::Active as usize * frontier_capacity + slot) = 1;
            }
        }
    }
}

const F32_SIGN_BIT: u32 = 1 << (u32::BITS - 1);

fn top_k_score_key(score: f32) -> u32 {
    let bits = score.to_bits();
    if bits & F32_SIGN_BIT == 0 {
        bits ^ F32_SIGN_BIT
    } else {
        !bits
    }
}

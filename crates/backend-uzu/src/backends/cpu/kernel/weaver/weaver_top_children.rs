use half::bf16;
use proc_macros::kernel;

use crate::{
    backends::common::gpu_types::weaver::{CANDIDATES_MAX, MetadataIdx},
    encodable_block::sampling::{gumbel_float, revidx},
};

#[kernel(WeaverTopChildren)]
pub fn weaver_top_children(
    residual_logits: *const bf16,
    candidate_logits: *const f32,
    candidate_ids: *const u32,
    depth_seeds: *const u64,
    node_metadata: *const u32,
    output_token_ids: *mut u32,
    output_model_logprobs: *mut f32,
    rows: u32,
    candidates: u32,
    children_per_node: u32,
    vocab_size: u32,
) {
    let rows = rows as usize;
    let candidates = candidates as usize;
    let children_per_node = children_per_node as usize;
    if candidates == 0 || candidates > CANDIDATES_MAX || children_per_node == 0 || children_per_node > candidates {
        return;
    }
    for row in 0..rows {
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
        let mut indices = (0..candidates).collect::<Vec<_>>();
        indices.sort_by(|&left, &right| {
            perturbed[right].total_cmp(&perturbed[left]).then_with(|| token(left).cmp(&token(right)))
        });
        for (rank, index) in indices.into_iter().take(children_per_node).enumerate() {
            unsafe {
                *output_token_ids.add(row * children_per_node + rank) = token(index);
                *output_model_logprobs.add(row * children_per_node + rank) = logits[index] - log_sum;
            }
        }
    }
}

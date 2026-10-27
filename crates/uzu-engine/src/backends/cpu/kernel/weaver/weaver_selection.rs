use std::range::Range;

use half::bf16;
use uzu_engine_macros::kernel;

use crate::{
    backends::common::gpu_types::weaver::{
        CANDIDATES_MAX, FRONTIER_MAX_SLOTS, FRONTIER_MAX_WIDTH, FRONTIER_NO_WINNER, FrontierIdx, MetadataIdx,
        PADDING_DEPTH, TreeIdx, WeaverSelectionMode,
    },
    encodable_block::sampling::{gumbel_float, revidx},
};

#[kernel(WeaverSelect)]
#[variants(BLOCK_THREADS, 256, 1024)]
pub fn weaver_select<const BLOCK_THREADS: u32>(
    #[optional(mode != WeaverSelectionMode::Frontier)] residual_logits: Option<*const bf16>,
    #[optional(mode != WeaverSelectionMode::Frontier)] depth_seeds: Option<*const u64>,
    frontier: *mut u32,
    packed_tree: *mut u32,
    #[optional(mode != WeaverSelectionMode::Children)] slot_ancestors: Option<*mut u32>,
    #[optional(mode != WeaverSelectionMode::Children)] node_token_ids: Option<*mut u32>,
    node_metadata: *mut u32,
    #[optional(mode != WeaverSelectionMode::Children)] node_ancestor_indices: Option<*mut u32>,
    node_valid: *mut u32,
    candidate_pool_ids: *const u32,
    candidate_pool_logits: *const f32,
    node_candidate_ids: *mut u32,
    node_candidate_logits: *mut f32,
    children_rows: u32,
    children_from_pool: u32,
    candidates: u32,
    expand_width: u32,
    vocab_size: u32,
    frontier_capacity: u32,
    tree_slot_count: u32,
    node_count: u32,
    batch_start_slot: u32,
    ancestor_stride: u32,
    lookahead_count: u32,
    candidate_depth_count: u32,
    #[optional(has_prune_noise)] prune_noise_scale: Option<f32>,
    #[specialize] mode: WeaverSelectionMode,
    #[specialize] has_prune_noise: bool,
) {
    if mode != WeaverSelectionMode::Frontier
        && candidates != 0
        && candidates <= CANDIDATES_MAX
        && expand_width != 0
        && expand_width <= candidates
        && frontier_capacity != 0
        && tree_slot_count != 0
    {
        let residual_logits = residual_logits.unwrap();
        let depth_seeds = depth_seeds.unwrap();
        let (candidate_logits, candidate_ids) = if children_from_pool != 0 {
            (candidate_pool_logits, candidate_pool_ids)
        } else {
            (node_candidate_logits as *const f32, node_candidate_ids as *const u32)
        };

        let rows = children_rows as usize;
        let candidates = candidates as usize;
        let expand_width = expand_width as usize;
        let frontier_capacity = frontier_capacity as usize;
        let tree_slot_count = tree_slot_count as usize;
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
                .map(|index| unsafe {
                    *candidate_logits.add(base + index) + (*residual_logits.add(base + index)).to_f32()
                })
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
                    let path_logprob = f32::from_bits(
                        *packed_tree.add(TreeIdx::PathLogprobBits as usize * tree_slot_count + tree_slot),
                    );
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
    if mode != WeaverSelectionMode::Children {
        let slot_ancestors = slot_ancestors.unwrap();
        let node_token_ids = node_token_ids.unwrap();
        let node_ancestor_indices = node_ancestor_indices.unwrap();
        if frontier_capacity == 0
            || frontier_capacity > FRONTIER_MAX_SLOTS
            || node_count == 0
            || node_count > FRONTIER_MAX_WIDTH
            || ancestor_stride == 0
            || tree_slot_count == 0
            || batch_start_slot + node_count > tree_slot_count
            || candidate_depth_count == 0
            || candidates == 0
        {
            return;
        }
        let (frontier_capacity, tree_slot_count, node_count, ancestor_stride) =
            (frontier_capacity as usize, tree_slot_count as usize, node_count as usize, ancestor_stride as usize);
        let (candidate_depth_count, candidates) = (candidate_depth_count as usize, candidates as usize);
        let pool_len = candidate_depth_count * candidates;
        let frontier = unsafe { std::slice::from_raw_parts_mut(frontier, FrontierIdx::COUNT * frontier_capacity) };
        let packed_tree = unsafe { std::slice::from_raw_parts_mut(packed_tree, TreeIdx::COUNT * tree_slot_count) };
        let slot_ancestors =
            unsafe { std::slice::from_raw_parts_mut(slot_ancestors, tree_slot_count * ancestor_stride) };
        let node_token_ids = unsafe { std::slice::from_raw_parts_mut(node_token_ids, node_count) };
        let node_metadata = unsafe { std::slice::from_raw_parts_mut(node_metadata, MetadataIdx::COUNT * node_count) };
        let node_ancestor_indices =
            unsafe { std::slice::from_raw_parts_mut(node_ancestor_indices, node_count * ancestor_stride) };
        let node_valid = unsafe { std::slice::from_raw_parts_mut(node_valid, node_count) };
        let candidate_pool_ids = unsafe { std::slice::from_raw_parts(candidate_pool_ids, pool_len) };
        let candidate_pool_logits = unsafe { std::slice::from_raw_parts(candidate_pool_logits, pool_len) };
        let node_candidate_ids = unsafe { std::slice::from_raw_parts_mut(node_candidate_ids, node_count * candidates) };
        let node_candidate_logits =
            unsafe { std::slice::from_raw_parts_mut(node_candidate_logits, node_count * candidates) };

        for node in 0..node_count {
            let (mut key, mut parent, mut token, mut winner) =
                (0, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER);
            for slot in 0..frontier_capacity {
                if frontier[FrontierIdx::Active as usize * frontier_capacity + slot] == 0 {
                    continue;
                }
                let score_key = frontier[FrontierIdx::PathScoreKey as usize * frontier_capacity + slot];
                let expandable =
                    u32::from(frontier[FrontierIdx::Depth as usize * frontier_capacity + slot] < lookahead_count);
                let next = (
                    (expandable << 31) | (score_key >> 1),
                    frontier[FrontierIdx::ParentSlot as usize * frontier_capacity + slot],
                    frontier[FrontierIdx::TokenId as usize * frontier_capacity + slot],
                );
                if next.0 > key || (next.0 == key && (next.1 < parent || (next.1 == parent && next.2 < token))) {
                    (key, parent, token, winner) = (next.0, next.1, next.2, slot as u32);
                }
            }
            let real = winner != FRONTIER_NO_WINNER;
            let winner = if real {
                winner as usize
            } else {
                0
            };
            let tree_slot = batch_start_slot as usize + node;
            let field_value =
                |field: FrontierIdx| u32::from(real) * frontier[field as usize * frontier_capacity + winner];
            let token = field_value(FrontierIdx::TokenId);
            let depth = field_value(FrontierIdx::Depth);
            let cumulative_logprob = field_value(FrontierIdx::PathLogprobBits);
            let logprob = field_value(FrontierIdx::EdgeLogprobBits);

            packed_tree[TreeIdx::TokenId as usize * tree_slot_count + tree_slot] = token;
            packed_tree[TreeIdx::ParentSlot as usize * tree_slot_count + tree_slot] = if real {
                parent
            } else {
                FRONTIER_NO_WINNER
            };
            packed_tree[TreeIdx::Depth as usize * tree_slot_count + tree_slot] = depth;
            packed_tree[TreeIdx::PathLogprobBits as usize * tree_slot_count + tree_slot] = cumulative_logprob;
            packed_tree[TreeIdx::EdgeLogprobBits as usize * tree_slot_count + tree_slot] = logprob;
            packed_tree[TreeIdx::Valid as usize * tree_slot_count + tree_slot] = u32::from(real);

            if real {
                frontier[FrontierIdx::Active as usize * frontier_capacity + winner] = 0;
            }

            let parent_slot = if real && (parent as usize) < tree_slot_count {
                parent as usize
            } else {
                0
            };
            for index in 0..ancestor_stride {
                let ancestor = if real && index + 1 < depth as usize {
                    slot_ancestors[parent_slot * ancestor_stride + index]
                } else if real && index + 1 == depth as usize {
                    parent_slot as u32
                } else {
                    0
                };
                slot_ancestors[tree_slot * ancestor_stride + index] = ancestor;
                node_ancestor_indices[node * ancestor_stride + index] = ancestor;
            }

            node_token_ids[node] = token;
            let expandable = depth < lookahead_count;
            node_metadata[MetadataIdx::Depth as usize * node_count + node] = if expandable {
                depth
            } else {
                PADDING_DEPTH
            };
            node_metadata[MetadataIdx::AncestorCount as usize * node_count + node] = depth;
            node_metadata[MetadataIdx::TreeSlot as usize * node_count + node] = tree_slot as u32;
            node_valid[node] = u32::from(real && expandable);

            if (depth as usize) < candidate_depth_count {
                let source = Range::from(depth as usize * candidates..(depth as usize + 1) * candidates);
                let destination = Range::from(node * candidates..(node + 1) * candidates);
                node_candidate_ids[destination].copy_from_slice(&candidate_pool_ids[source]);
                node_candidate_logits[destination].copy_from_slice(&candidate_pool_logits[source]);
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

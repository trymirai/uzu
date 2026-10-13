use half::bf16;
use uzu_engine_macros::kernel;

use super::math::{Dims, alpha_beta, child_edges, edge_delta, log_softmax_floor, pool_summary, slot_numbers};
use crate::{
    backends::common::gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx},
    encodable_block::confidence_head::layout::{LocalLayout, SLOT_NUMBERS},
};

/// The head's local part on the nodes one round expanded, right after Weaver's top_children put their children into
/// the frontier (entry slot * children + j for child j of the node in tree slot `slot`): for every child the edge
/// min(log q_u(c), 0), which replaces the model's in the frontier entry, so that its path is the parent's plus the
/// edge (log r) and its priority follows; and the node's rows of the step state the final prune reads (z_u, every
/// child's token vector and slot numbers, its pool index and whether it is real, the pool logits l = a + rho).
#[kernel(ConfidenceHeadRound)]
pub fn confidence_head_round(
    residual_logits: *const bf16,
    candidate_logits: *const f32,
    candidate_ids: *const u32,
    noise: *const f32,
    draft_rows: *const f32,
    node_metadata: *const u32,
    node_valid: *const u32,
    packed_tree: *const u32,
    token_table: *const bf16,
    local_weights: *const f32,
    child_edges_out: *mut f32,
    frontier: *mut u32,
    nodes: *mut f32,
    child_inputs: *mut f32,
    child_indices: *mut u32,
    child_real: *mut u32,
    slot_logits: *mut f32,
    rows: u32,
    pool_rows: u32,
    cap: f32,
    epsilon: f32,
    frontier_capacity: u32,
    tree_slot_count: u32,
    #[specialize] candidates: u32,
    #[specialize] children: u32,
    #[specialize] hidden: u32,
    #[specialize] pool_hidden: u32,
    #[specialize] edge_hidden: u32,
) {
    let (rows, candidates, children) = (rows as usize, candidates as usize, children as usize);
    let (frontier_capacity, tree_slot_count) = (frontier_capacity as usize, tree_slot_count as usize);
    let dims = Dims {
        hidden: hidden as usize,
        pool_hidden: pool_hidden as usize,
        edge_hidden: edge_hidden as usize,
    };
    let layout = LocalLayout::new(dims.hidden, dims.pool_hidden, dims.edge_hidden);
    let weights = unsafe { std::slice::from_raw_parts(local_weights, layout.total) };
    let hidden = dims.hidden;
    let input_width = hidden + SLOT_NUMBERS;
    for row in 0..rows {
        if unsafe { *node_valid.add(row) } == 0 {
            continue;
        }
        let depth = unsafe { *node_metadata.add(MetadataIdx::Depth as usize * rows + row) } as usize;
        let slot = unsafe { *node_metadata.add(MetadataIdx::TreeSlot as usize * rows + row) } as usize;
        let pool_row = depth.min(pool_rows as usize - 1);
        let draft =
            (0..candidates).map(|index| unsafe { *candidate_logits.add(row * candidates + index) }).collect::<Vec<_>>();
        let logits = (0..candidates)
            .map(|index| draft[index] + unsafe { (*residual_logits.add(row * candidates + index)).to_f32() })
            .collect::<Vec<_>>();
        let tokens =
            (0..candidates).map(|index| unsafe { *candidate_ids.add(row * candidates + index) }).collect::<Vec<_>>();
        let gumbels =
            (0..candidates).map(|index| unsafe { *noise.add(pool_row * candidates + index) }).collect::<Vec<_>>();
        let log_draft = log_softmax_floor(&draft);
        let log_logits = log_softmax_floor(&logits);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = logits.iter().map(|value| (value - max).exp()).sum::<f32>();
        let weight = logits.iter().map(|value| (value - max).exp() / sum).collect::<Vec<_>>();
        let summary = pool_summary(weights, &layout, &dims, &log_draft, &log_logits, &gumbels, &weight);
        let node = (0..hidden)
            .map(|unit| unsafe { *draft_rows.add(pool_row * hidden + unit) } + summary[unit])
            .collect::<Vec<_>>();
        let (alpha, beta) = alpha_beta(weights, &layout, &node, cap);

        let mut indices = vec![0usize; children];
        let mut deltas = vec![0.0f32; children];
        for child in 0..children {
            let token =
                unsafe { *frontier.add(FrontierIdx::TokenId as usize * frontier_capacity + slot * children + child) };
            // Pool tokens are distinct: a child is found once.
            let index = tokens.iter().position(|&candidate| candidate == token).unwrap_or(0);
            let vector_token = tokens[index] as usize;
            let vector = (0..hidden)
                .map(|unit| unsafe { (*token_table.add(vector_token * hidden + unit)).to_f32() })
                .collect::<Vec<_>>();
            let rank = draft.iter().filter(|&&value| value > draft[index]).count() as f32;
            let numbers = slot_numbers(
                weights,
                &layout,
                [log_draft[index], log_logits[index], gumbels[index], (1.0 + rank).ln()],
            );
            deltas[child] = edge_delta(weights, &layout, &dims, &node, &vector, &numbers, epsilon);
            indices[child] = index;
            let kept = slot * children + child;
            unsafe {
                for unit in 0..hidden {
                    *child_inputs.add(kept * input_width + unit) = vector[unit];
                }
                for (offset, number) in numbers.iter().enumerate() {
                    *child_inputs.add(kept * input_width + hidden + offset) = *number;
                }
                *child_indices.add(kept) = index as u32;
                *child_real.add(kept) = 1;
            }
        }
        let edges = child_edges(alpha, beta, &logits, &gumbels, &indices, &deltas, &vec![true; children]);
        let parent_path =
            f32::from_bits(unsafe { *packed_tree.add(TreeIdx::PathLogprobBits as usize * tree_slot_count + slot) });
        unsafe {
            for (child, edge) in edges.into_iter().enumerate() {
                let entry = slot * children + child;
                let path = parent_path + edge;
                *child_edges_out.add(row * children + child) = edge;
                *frontier.add(FrontierIdx::PathLogprobBits as usize * frontier_capacity + entry) = path.to_bits();
                *frontier.add(FrontierIdx::EdgeLogprobBits as usize * frontier_capacity + entry) = edge.to_bits();
                *frontier.add(FrontierIdx::PathScoreKey as usize * frontier_capacity + entry) = score_key(path);
            }
            for (unit, value) in node.iter().enumerate() {
                *nodes.add(slot * hidden + unit) = *value;
            }
            for (index, value) in logits.iter().enumerate() {
                *slot_logits.add(slot * candidates + index) = *value;
            }
        }
    }
}

/// The frontier's priority key of a path log-probability: larger keys follow the f32 order (as `top_k_score_key` of
/// the Metal kernels).
fn score_key(score: f32) -> u32 {
    let bits = score.to_bits();
    if bits & (1 << 31) == 0 {
        bits ^ (1 << 31)
    } else {
        !bits
    }
}

use uzu_engine_macros::kernel;

use super::math::{Dims, alpha_beta, child_edges, edge_delta};
use crate::{
    backends::common::gpu_types::weaver::{FrontierIdx, TreeIdx},
    encodable_block::confidence_head::layout::{LocalLayout, SLOT_NUMBERS},
};

const NONE: u32 = u32::MAX;

/// The full head's edge of every real child: the slot's z_u and the children's token vectors corrected by the tree
/// attention (node_shifts at the slot's token, child_shifts at each child's token), then alpha, beta and delta as
/// in the local part and log q over the slot's pool (its kept logits, the pool row's noise). min(log q, 0) goes to
/// the edge lane the final prune sums along paths: the packed tree's for a child expanded into a slot, the
/// frontier's for one still in the frontier.
#[kernel(ConfidenceHeadPrune)]
pub fn confidence_head_prune(
    nodes: *const f32,
    child_inputs: *const f32,
    child_indices: *const u32,
    child_real: *const u32,
    slot_logits: *const f32,
    noise: *const f32,
    node_shifts: *const f32,
    child_shifts: *const f32,
    slot_tokens: *const u32,
    token_slots: *const u32,
    local_weights: *const f32,
    packed_tree: *mut u32,
    frontier: *mut u32,
    slot_count: u32,
    pool_rows: u32,
    cap: f32,
    epsilon: f32,
    #[specialize] candidates: u32,
    #[specialize] children: u32,
    #[specialize] hidden: u32,
    #[specialize] pool_hidden: u32,
    #[specialize] edge_hidden: u32,
) {
    let (slots, candidates, children) = (slot_count as usize, candidates as usize, children as usize);
    let dims = Dims {
        hidden: hidden as usize,
        pool_hidden: pool_hidden as usize,
        edge_hidden: edge_hidden as usize,
    };
    let hidden = dims.hidden;
    let layout = LocalLayout::new(hidden, dims.pool_hidden, dims.edge_hidden);
    let weights = unsafe { std::slice::from_raw_parts(local_weights, layout.total) };
    let input_width = hidden + SLOT_NUMBERS;
    let entries = slots * children;
    for slot in 0..slots {
        let real =
            (0..children).map(|child| unsafe { *child_real.add(slot * children + child) } != 0).collect::<Vec<_>>();
        if !real.iter().any(|&is_real| is_real) {
            continue;
        }
        let depth = unsafe { *packed_tree.add(TreeIdx::Depth as usize * slots + slot) } as usize;
        let pool_row = depth.min(pool_rows as usize - 1);
        let token = unsafe { *slot_tokens.add(slot) } as usize;
        let node = (0..hidden)
            .map(|unit| unsafe { *nodes.add(slot * hidden + unit) + *node_shifts.add(token * hidden + unit) })
            .collect::<Vec<_>>();
        let (alpha, beta) = alpha_beta(weights, &layout, &node, cap);
        let logits = unsafe { std::slice::from_raw_parts(slot_logits.add(slot * candidates), candidates) };
        let gumbels = unsafe { std::slice::from_raw_parts(noise.add(pool_row * candidates), candidates) };
        let mut indices = vec![0usize; children];
        let mut deltas = vec![0.0f32; children];
        for child in 0..children {
            let kept = slot * children + child;
            indices[child] = unsafe { *child_indices.add(kept) } as usize;
            if !real[child] {
                continue;
            }
            let child_token = 1 + kept;
            let vector = (0..hidden)
                .map(|unit| unsafe {
                    *child_inputs.add(kept * input_width + unit) + *child_shifts.add(child_token * hidden + unit)
                })
                .collect::<Vec<_>>();
            let numbers =
                unsafe { std::slice::from_raw_parts(child_inputs.add(kept * input_width + hidden), SLOT_NUMBERS) };
            deltas[child] = edge_delta(weights, &layout, &dims, &node, &vector, numbers, epsilon);
        }
        let edges = child_edges(alpha, beta, logits, gumbels, &indices, &deltas, &real);
        for child in 0..children {
            if !real[child] {
                continue;
            }
            let kept = slot * children + child;
            let bits = edges[child].to_bits();
            let expanded = unsafe { *token_slots.add(1 + kept) };
            unsafe {
                if expanded != NONE {
                    *packed_tree.add(TreeIdx::EdgeLogprobBits as usize * slots + expanded as usize) = bits;
                } else {
                    *frontier.add(FrontierIdx::EdgeLogprobBits as usize * entries + kept) = bits;
                }
            }
        }
    }
}

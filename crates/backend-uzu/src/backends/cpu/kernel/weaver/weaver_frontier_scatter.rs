use proc_macros::kernel;

use crate::backends::common::gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx};

const F32_SIGN_BIT: u32 = 1 << (u32::BITS - 1);

fn top_k_score_key(score: f32) -> u32 {
    let bits = score.to_bits();
    if bits & F32_SIGN_BIT == 0 {
        bits ^ F32_SIGN_BIT
    } else {
        !bits
    }
}

#[kernel(WeaverFrontierScatter)]
pub fn weaver_frontier_scatter(
    tree: *const u32,
    round_metadata: *const u32,
    round_valid: *const u32,
    child_ids: *const u32,
    child_logprobs: *const f32,
    frontier: *mut u32,
    capacity: u32,
    tree_slots: u32,
    rows: u32,
    children_per_node: u32,
) {
    if capacity == 0 || tree_slots == 0 || children_per_node == 0 {
        return;
    }
    let (capacity, tree_slots, rows, children_per_node) =
        (capacity as usize, tree_slots as usize, rows as usize, children_per_node as usize);
    let tree = unsafe { std::slice::from_raw_parts(tree, TreeIdx::COUNT * tree_slots) };
    let parent_indices =
        unsafe { std::slice::from_raw_parts(round_metadata.add(MetadataIdx::TreeSlot as usize * rows), rows) };
    let round_valid = unsafe { std::slice::from_raw_parts(round_valid, rows) };
    let child_ids = unsafe { std::slice::from_raw_parts(child_ids, rows * children_per_node) };
    let child_logprobs = unsafe { std::slice::from_raw_parts(child_logprobs, rows * children_per_node) };
    let frontier = unsafe { std::slice::from_raw_parts_mut(frontier, FrontierIdx::COUNT * capacity) };

    for index in 0..rows * children_per_node {
        let row = index / children_per_node;
        if round_valid[row] == 0 {
            continue;
        }
        let parent = parent_indices[row] as usize;
        let slot = parent * children_per_node + index % children_per_node;
        if parent >= tree_slots || slot >= capacity {
            continue;
        }
        let logprob = child_logprobs[index];
        let cumulative_logprob =
            f32::from_bits(tree[TreeIdx::PathLogprobBits as usize * tree_slots + parent]) + logprob;
        let mut set = |field: FrontierIdx, value| {
            frontier[field as usize * capacity + slot] = value;
        };
        set(FrontierIdx::TokenId, child_ids[index]);
        set(FrontierIdx::ParentSlot, parent as u32);
        set(FrontierIdx::Depth, tree[TreeIdx::Depth as usize * tree_slots + parent] + 1);
        set(FrontierIdx::PathLogprobBits, cumulative_logprob.to_bits());
        set(FrontierIdx::EdgeLogprobBits, logprob.to_bits());
        set(FrontierIdx::PathScoreKey, top_k_score_key(cumulative_logprob));
        set(FrontierIdx::Active, 1);
    }
}

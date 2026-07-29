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

#[kernel(WeaverFrontierInsertChildren)]
pub fn weaver_frontier_insert_children(
    packed_tree: *const u32,
    node_metadata: *const u32,
    node_valid: *const u32,
    child_ids: *const u32,
    child_logprobs: *const f32,
    frontier: *mut u32,
    frontier_capacity: u32,
    tree_slot_count: u32,
    node_count: u32,
    children_per_node: u32,
) {
    if frontier_capacity == 0 || tree_slot_count == 0 || children_per_node == 0 {
        return;
    }
    let (frontier_capacity, tree_slot_count, node_count, children_per_node) =
        (frontier_capacity as usize, tree_slot_count as usize, node_count as usize, children_per_node as usize);
    let packed_tree = unsafe { std::slice::from_raw_parts(packed_tree, TreeIdx::COUNT * tree_slot_count) };
    let parent_indices = unsafe {
        std::slice::from_raw_parts(node_metadata.add(MetadataIdx::TreeSlot as usize * node_count), node_count)
    };
    let node_valid = unsafe { std::slice::from_raw_parts(node_valid, node_count) };
    let child_ids = unsafe { std::slice::from_raw_parts(child_ids, node_count * children_per_node) };
    let child_logprobs = unsafe { std::slice::from_raw_parts(child_logprobs, node_count * children_per_node) };
    let frontier = unsafe { std::slice::from_raw_parts_mut(frontier, FrontierIdx::COUNT * frontier_capacity) };

    for index in 0..node_count * children_per_node {
        let row = index / children_per_node;
        if node_valid[row] == 0 {
            continue;
        }
        let parent = parent_indices[row] as usize;
        let slot = parent * children_per_node + index % children_per_node;
        if parent >= tree_slot_count || slot >= frontier_capacity {
            continue;
        }
        let logprob = child_logprobs[index];
        let cumulative_logprob =
            f32::from_bits(packed_tree[TreeIdx::PathLogprobBits as usize * tree_slot_count + parent]) + logprob;
        frontier[FrontierIdx::TokenId as usize * frontier_capacity + slot] = child_ids[index];
        frontier[FrontierIdx::ParentSlot as usize * frontier_capacity + slot] = parent as u32;
        frontier[FrontierIdx::Depth as usize * frontier_capacity + slot] =
            packed_tree[TreeIdx::Depth as usize * tree_slot_count + parent] + 1;
        frontier[FrontierIdx::PathLogprobBits as usize * frontier_capacity + slot] = cumulative_logprob.to_bits();
        frontier[FrontierIdx::EdgeLogprobBits as usize * frontier_capacity + slot] = logprob.to_bits();
        frontier[FrontierIdx::PathScoreKey as usize * frontier_capacity + slot] = top_k_score_key(cumulative_logprob);
        frontier[FrontierIdx::Active as usize * frontier_capacity + slot] = 1;
    }
}

use proc_macros::kernel;

use crate::backends::common::gpu_types::weaver::{
    FRONTIER_MAX_SLOTS, FRONTIER_MAX_WIDTH, FRONTIER_NO_WINNER, FrontierIdx, MetadataIdx, TreeIdx,
};

#[kernel(WeaverFrontierSelect)]
pub fn weaver_frontier_select(
    frontier: *mut u32,
    packed_tree: *mut u32,
    slot_ancestors: *mut u32,
    node_token_ids: *mut u32,
    node_metadata: *mut u32,
    node_ancestor_indices: *mut u32,
    node_valid: *mut u32,
    candidate_pool_ids: *const u32,
    candidate_pool_logits: *const f32,
    node_candidate_ids: *mut u32,
    node_candidate_logits: *mut f32,
    frontier_capacity: u32,
    tree_slot_count: u32,
    node_count: u32,
    batch_start_slot: u32,
    ancestor_stride: u32,
    max_depth: u32,
    lookahead_count: u32,
    candidate_depth_count: u32,
    candidates_per_depth: u32,
) {
    if frontier_capacity == 0
        || frontier_capacity as usize > FRONTIER_MAX_SLOTS
        || node_count == 0
        || node_count as usize > FRONTIER_MAX_WIDTH
        || ancestor_stride == 0
        || max_depth == 0
        || tree_slot_count == 0
        || batch_start_slot + node_count > tree_slot_count
        || candidate_depth_count == 0
        || candidates_per_depth == 0
    {
        return;
    }
    let (frontier_capacity, tree_slot_count, node_count, ancestor_stride) =
        (frontier_capacity as usize, tree_slot_count as usize, node_count as usize, ancestor_stride as usize);
    let (candidate_depth_count, candidates_per_depth) = (candidate_depth_count as usize, candidates_per_depth as usize);
    let pool_len = candidate_depth_count * candidates_per_depth;
    let frontier = unsafe { std::slice::from_raw_parts_mut(frontier, FrontierIdx::COUNT * frontier_capacity) };
    let packed_tree = unsafe { std::slice::from_raw_parts_mut(packed_tree, TreeIdx::COUNT * tree_slot_count) };
    let slot_ancestors = unsafe { std::slice::from_raw_parts_mut(slot_ancestors, tree_slot_count * ancestor_stride) };
    let node_token_ids = unsafe { std::slice::from_raw_parts_mut(node_token_ids, node_count) };
    let node_metadata = unsafe { std::slice::from_raw_parts_mut(node_metadata, MetadataIdx::COUNT * node_count) };
    let node_ancestor_indices =
        unsafe { std::slice::from_raw_parts_mut(node_ancestor_indices, node_count * ancestor_stride) };
    let node_valid = unsafe { std::slice::from_raw_parts_mut(node_valid, node_count) };
    let candidate_pool_ids = unsafe { std::slice::from_raw_parts(candidate_pool_ids, pool_len) };
    let candidate_pool_logits = unsafe { std::slice::from_raw_parts(candidate_pool_logits, pool_len) };
    let node_candidate_ids =
        unsafe { std::slice::from_raw_parts_mut(node_candidate_ids, node_count * candidates_per_depth) };
    let node_candidate_logits =
        unsafe { std::slice::from_raw_parts_mut(node_candidate_logits, node_count * candidates_per_depth) };

    for node in 0..node_count {
        let (mut key, mut parent, mut token, mut winner) =
            (0, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER);
        for slot in 0..frontier_capacity {
            if frontier[FrontierIdx::Active as usize * frontier_capacity + slot] == 0 {
                continue;
            }
            let next = (
                frontier[FrontierIdx::PathScoreKey as usize * frontier_capacity + slot],
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
        let field_value = |field: FrontierIdx| u32::from(real) * frontier[field as usize * frontier_capacity + winner];
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
        node_metadata[MetadataIdx::Depth as usize * node_count + node] = depth.min(max_depth - 1);
        node_metadata[MetadataIdx::AncestorCount as usize * node_count + node] = depth.min(ancestor_stride as u32);
        node_metadata[MetadataIdx::TreeSlot as usize * node_count + node] = tree_slot as u32;
        node_valid[node] = u32::from(real && depth < lookahead_count && depth < max_depth);

        // Every node expands the candidate pool for its own depth.
        let candidate_depth = (depth as usize).min(candidate_depth_count - 1);
        let source = candidate_depth * candidates_per_depth..(candidate_depth + 1) * candidates_per_depth;
        let destination = node * candidates_per_depth..(node + 1) * candidates_per_depth;
        node_candidate_ids[destination.clone()].copy_from_slice(&candidate_pool_ids[source.clone()]);
        node_candidate_logits[destination].copy_from_slice(&candidate_pool_logits[source]);
    }
}

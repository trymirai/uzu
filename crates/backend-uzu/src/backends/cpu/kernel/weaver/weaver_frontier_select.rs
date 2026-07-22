use proc_macros::kernel;

use crate::backends::common::kernel::weaver::*;

#[kernel(WeaverFrontierSelect)]
pub fn weaver_frontier_select(
    frontier: *mut u32,
    tree: *mut u32,
    slot_ancestors: *mut u32,
    round_token_ids: *mut u32,
    round_metadata: *mut u32,
    round_ancestors: *mut u32,
    round_valid: *mut u32,
    candidate_pool_ids: *const u32,
    candidate_pool_scores: *const f32,
    round_candidate_ids: *mut u32,
    round_candidate_scores: *mut f32,
    capacity: u32,
    tree_slots: u32,
    width: u32,
    slot_start: u32,
    ancestor_stride: u32,
    max_depth: u32,
    lookahead_count: u32,
    candidate_pool_rows: u32,
    candidate_pool_size: u32,
) {
    if capacity == 0
        || capacity as usize > MAX_FRONTIER_SLOTS
        || width == 0
        || width as usize > MAX_FRONTIER_WIDTH
        || ancestor_stride == 0
        || max_depth == 0
        || tree_slots == 0
        || slot_start + width > tree_slots
        || candidate_pool_rows == 0
        || candidate_pool_size == 0
    {
        return;
    }
    let (capacity, slots, width, stride) =
        (capacity as usize, tree_slots as usize, width as usize, ancestor_stride as usize);
    let (candidate_pool_rows, candidate_pool_size) = (candidate_pool_rows as usize, candidate_pool_size as usize);
    let pool_len = candidate_pool_rows * candidate_pool_size;
    let frontier = unsafe { std::slice::from_raw_parts_mut(frontier, FRONTIER_LANE_COUNT * capacity) };
    let tree = unsafe { std::slice::from_raw_parts_mut(tree, TREE_LANE_COUNT * slots) };
    let slot_ancestors = unsafe { std::slice::from_raw_parts_mut(slot_ancestors, slots * stride) };
    let round_token_ids = unsafe { std::slice::from_raw_parts_mut(round_token_ids, width) };
    let round_metadata = unsafe { std::slice::from_raw_parts_mut(round_metadata, METADATA_LANE_COUNT * width) };
    let round_ancestors = unsafe { std::slice::from_raw_parts_mut(round_ancestors, width * stride) };
    let round_valid = unsafe { std::slice::from_raw_parts_mut(round_valid, width) };
    let candidate_pool_ids = unsafe { std::slice::from_raw_parts(candidate_pool_ids, pool_len) };
    let candidate_pool_scores = unsafe { std::slice::from_raw_parts(candidate_pool_scores, pool_len) };
    let round_candidate_ids =
        unsafe { std::slice::from_raw_parts_mut(round_candidate_ids, width * candidate_pool_size) };
    let round_candidate_scores =
        unsafe { std::slice::from_raw_parts_mut(round_candidate_scores, width * candidate_pool_size) };

    for row in 0..width {
        let (mut key, mut parent, mut token, mut winner) = (0, NO_WINNER, NO_WINNER, NO_WINNER);
        for slot in 0..capacity {
            if frontier[FRONTIER_LANE_ACTIVE * capacity + slot] == 0 {
                continue;
            }
            let next = (
                frontier[FRONTIER_LANE_KEY * capacity + slot],
                frontier[FRONTIER_LANE_PARENT * capacity + slot],
                frontier[FRONTIER_LANE_TOKEN * capacity + slot],
            );
            if next.0 > key || (next.0 == key && (next.1 < parent || (next.1 == parent && next.2 < token))) {
                (key, parent, token, winner) = (next.0, next.1, next.2, slot as u32);
            }
        }
        let real = winner != NO_WINNER;
        let winner = if real {
            winner as usize
        } else {
            0
        };
        let tree_slot = slot_start as usize + row;
        let lane_value = |lane: usize| u32::from(real) * frontier[lane * capacity + winner];
        let token = lane_value(FRONTIER_LANE_TOKEN);
        let depth = lane_value(FRONTIER_LANE_DEPTH);
        let cumulative_logprob = lane_value(FRONTIER_LANE_CUM);
        let logprob = lane_value(FRONTIER_LANE_LOGPROB);

        tree[TREE_LANE_TOKEN * slots + tree_slot] = token;
        tree[TREE_LANE_PARENT * slots + tree_slot] = if real {
            parent
        } else {
            NO_WINNER
        };
        tree[TREE_LANE_DEPTH * slots + tree_slot] = depth;
        tree[TREE_LANE_CUM * slots + tree_slot] = cumulative_logprob;
        tree[TREE_LANE_LOGPROB * slots + tree_slot] = logprob;
        tree[TREE_LANE_MASK * slots + tree_slot] = u32::from(real);

        if real {
            frontier[FRONTIER_LANE_ACTIVE * capacity + winner] = 0;
        }

        let parent_slot = if real && (parent as usize) < slots {
            parent as usize
        } else {
            0
        };
        for index in 0..stride {
            let ancestor = if real && index + 1 < depth as usize {
                slot_ancestors[parent_slot * stride + index]
            } else if real && index + 1 == depth as usize {
                parent_slot as u32
            } else {
                0
            };
            slot_ancestors[tree_slot * stride + index] = ancestor;
            round_ancestors[row * stride + index] = ancestor;
        }

        round_token_ids[row] = token;
        round_metadata[METADATA_LANE_DEPTH * width + row] = depth.min(max_depth - 1);
        round_metadata[METADATA_LANE_ANCESTOR_COUNT * width + row] = depth.min(stride as u32);
        round_metadata[METADATA_LANE_NODE_INDEX * width + row] = tree_slot as u32;
        round_valid[row] = u32::from(real && depth < lookahead_count && depth < max_depth);

        // Every row of a round expands the pool for its own depth, so the winner's pool row is
        // copied into the contiguous per-round buffer the Weaver step reads.
        let candidate_pool_row = (depth as usize).min(candidate_pool_rows - 1);
        let source = candidate_pool_row * candidate_pool_size..(candidate_pool_row + 1) * candidate_pool_size;
        let destination = row * candidate_pool_size..(row + 1) * candidate_pool_size;
        round_candidate_ids[destination.clone()].copy_from_slice(&candidate_pool_ids[source.clone()]);
        round_candidate_scores[destination].copy_from_slice(&candidate_pool_scores[source]);
    }
}

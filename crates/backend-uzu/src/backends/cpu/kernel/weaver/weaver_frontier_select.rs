use proc_macros::kernel;

use crate::backends::common::kernel::weaver::*;

#[kernel(WeaverFrontierSelect)]
#[allow(clippy::too_many_arguments)]
pub fn weaver_frontier_select(
    frontier: *mut u32,
    tree: *mut u32,
    slot_ancestors: *mut u32,
    round_token: *mut u32,
    round_metadata: *mut u32,
    round_ancestors: *mut u32,
    round_valid: *mut u32,
    capacity: u32,
    tree_slots: u32,
    width: u32,
    slot_start: u32,
    ancestor_stride: u32,
    max_depth: u32,
    lookahead_count: u32,
) {
    if capacity == 0
        || capacity as usize > MAX_FRONTIER_SLOTS
        || width == 0
        || width as usize > MAX_FRONTIER_WIDTH
        || ancestor_stride == 0
        || max_depth == 0
        || tree_slots == 0
        || slot_start + width > tree_slots
    {
        return;
    }
    let (capacity, slots, width, stride) =
        (capacity as usize, tree_slots as usize, width as usize, ancestor_stride as usize);
    let lane = |lane: usize, slot: usize| unsafe { *frontier.add(lane * capacity + slot) };
    for row in 0..width {
        let (mut key, mut parent, mut token, mut winner) = (0, NO_WINNER, NO_WINNER, NO_WINNER);
        for slot in 0..capacity {
            if lane(FRONTIER_LANE_ACTIVE, slot) == 0 {
                continue;
            }
            let next =
                (lane(FRONTIER_LANE_KEY, slot), lane(FRONTIER_LANE_PARENT, slot), lane(FRONTIER_LANE_TOKEN, slot));
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
        let value = |lane_index| u32::from(real) * lane(lane_index, winner);
        let token = value(FRONTIER_LANE_TOKEN);
        let depth = value(FRONTIER_LANE_DEPTH);
        unsafe {
            *tree.add(TREE_LANE_TOKEN * slots + tree_slot) = token;
            *tree.add(TREE_LANE_PARENT * slots + tree_slot) = if real {
                parent
            } else {
                NO_WINNER
            };
            *tree.add(TREE_LANE_DEPTH * slots + tree_slot) = depth;
            *tree.add(TREE_LANE_CUM * slots + tree_slot) = value(FRONTIER_LANE_CUM);
            *tree.add(TREE_LANE_LOGPROB * slots + tree_slot) = value(FRONTIER_LANE_LOGPROB);
            *tree.add(TREE_LANE_MASK * slots + tree_slot) = u32::from(real);
        }
        if real {
            unsafe { *frontier.add(FRONTIER_LANE_ACTIVE * capacity + winner) = 0 };
        }
        let parent = if real && (parent as usize) < slots {
            parent as usize
        } else {
            0
        };
        for index in 0..stride {
            let ancestor = if real && index + 1 < depth as usize {
                unsafe { *slot_ancestors.add(parent * stride + index) }
            } else if real && index + 1 == depth as usize {
                parent as u32
            } else {
                0
            };
            unsafe {
                *slot_ancestors.add(tree_slot * stride + index) = ancestor;
                *round_ancestors.add(row * stride + index) = ancestor;
            }
        }
        unsafe {
            *round_token.add(row) = token;
            *round_metadata.add(METADATA_LANE_DEPTH * width + row) = depth.min(max_depth - 1);
            *round_metadata.add(METADATA_LANE_ANCESTOR_COUNT * width + row) = depth.min(stride as u32);
            *round_metadata.add(METADATA_LANE_NODE_INDEX * width + row) = tree_slot as u32;
            *round_valid.add(row) = u32::from(real && depth < lookahead_count && depth < max_depth);
        }
    }
}

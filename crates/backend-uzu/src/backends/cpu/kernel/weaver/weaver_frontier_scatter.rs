use proc_macros::kernel;

use crate::backends::common::kernel::weaver::*;

fn top_k_score_key(score: f32) -> u32 {
    let bits = score.to_bits();
    if bits >> 31 == 0 {
        bits ^ 0x8000_0000
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
    fanout: u32,
) {
    if capacity == 0 || tree_slots == 0 || fanout == 0 {
        return;
    }
    unsafe {
        let (capacity, tree_slots, rows, fanout) =
            (capacity as usize, tree_slots as usize, rows as usize, fanout as usize);
        for index in 0..rows * fanout {
            let row = index / fanout;
            if *round_valid.add(row) == 0 {
                continue;
            }
            let parent = *round_metadata.add(METADATA_LANE_NODE_INDEX * rows + row) as usize;
            let slot = parent * fanout + index % fanout;
            if parent >= tree_slots || slot >= capacity {
                continue;
            }
            let logprob = *child_logprobs.add(index);
            let cum = f32::from_bits(*tree.add(TREE_LANE_CUM * tree_slots + parent)) + logprob;
            let values = [
                *child_ids.add(index),
                parent as u32,
                *tree.add(TREE_LANE_DEPTH * tree_slots + parent) + 1,
                cum.to_bits(),
                logprob.to_bits(),
                top_k_score_key(cum),
                1,
            ];
            for (lane, value) in values.into_iter().enumerate() {
                *frontier.add(lane * capacity + slot) = value;
            }
        }
    }
}

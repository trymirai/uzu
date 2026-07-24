use proc_macros::kernel;

use crate::backends::common::gpu_types::weaver;

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
    fanout: u32,
) {
    if capacity == 0 || tree_slots == 0 || fanout == 0 {
        return;
    }
    let (capacity, tree_slots, rows, fanout) = (capacity as usize, tree_slots as usize, rows as usize, fanout as usize);
    let tree = unsafe { std::slice::from_raw_parts(tree, weaver::TREE_LANE_COUNT * tree_slots) };
    let parent_indices =
        unsafe { std::slice::from_raw_parts(round_metadata.add(weaver::METADATA_LANE_NODE_INDEX * rows), rows) };
    let round_valid = unsafe { std::slice::from_raw_parts(round_valid, rows) };
    let child_ids = unsafe { std::slice::from_raw_parts(child_ids, rows * fanout) };
    let child_logprobs = unsafe { std::slice::from_raw_parts(child_logprobs, rows * fanout) };
    let frontier = unsafe { std::slice::from_raw_parts_mut(frontier, weaver::FRONTIER_LANE_COUNT * capacity) };

    for index in 0..rows * fanout {
        let row = index / fanout;
        if round_valid[row] == 0 {
            continue;
        }
        let parent = parent_indices[row] as usize;
        let slot = parent * fanout + index % fanout;
        if parent >= tree_slots || slot >= capacity {
            continue;
        }
        let logprob = child_logprobs[index];
        let cumulative_logprob = f32::from_bits(tree[weaver::TREE_LANE_CUM * tree_slots + parent]) + logprob;
        let values = [
            child_ids[index],
            parent as u32,
            tree[weaver::TREE_LANE_DEPTH * tree_slots + parent] + 1,
            cumulative_logprob.to_bits(),
            logprob.to_bits(),
            top_k_score_key(cumulative_logprob),
            1,
        ];
        for (lane, value) in values.into_iter().enumerate() {
            frontier[lane * capacity + slot] = value;
        }
    }
}

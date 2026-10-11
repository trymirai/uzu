use uzu_engine_macros::kernel;

use crate::backends::common::gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx};

const F32_SIGN_BIT: u32 = 1 << (u32::BITS - 1);

fn top_k_score_key(bits: u32) -> u32 {
    if bits & F32_SIGN_BIT == 0 {
        bits ^ F32_SIGN_BIT
    } else {
        !bits
    }
}

// These bits become an integer priority. Preserve the measured ARM CPU propagation explicitly instead of letting
// the compiler choose a NaN operand, sign or payload: signaling child/parent, then quiet child/parent.
fn path_logprob_bits(
    parent: u32,
    child: u32,
) -> u32 {
    let is_nan = |bits: u32| bits & 0x7fff_ffff > 0x7f80_0000;
    let is_signaling = |bits: u32| is_nan(bits) && bits & 0x0040_0000 == 0;
    for bits in [child, parent] {
        if is_signaling(bits) {
            return bits | 0x0040_0000;
        }
    }
    for bits in [child, parent] {
        if is_nan(bits) {
            return bits | 0x0040_0000;
        }
    }
    if parent & 0x7fff_ffff == 0x7f80_0000 && child == parent ^ F32_SIGN_BIT {
        return 0x7fc0_0000;
    }
    (f32::from_bits(parent) + f32::from_bits(child)).to_bits()
}

#[kernel(WeaverFrontierInsertChildren)]
pub fn weaver_frontier_insert_children(
    packed_tree: *const u32,
    node_metadata: *const u32,
    node_valid: *const u32,
    child_ids: *const u32,
    child_logprobs: *const f32,
    child_prune_logprobs: *const f32,
    frontier: *mut u32,
    frontier_capacity: u32,
    tree_slot_count: u32,
    node_count: u32,
    expand_width: u32,
) {
    if frontier_capacity == 0 || tree_slot_count == 0 || expand_width == 0 {
        return;
    }
    let (frontier_capacity, tree_slot_count, node_count, expand_width) =
        (frontier_capacity as usize, tree_slot_count as usize, node_count as usize, expand_width as usize);
    let packed_tree = unsafe { std::slice::from_raw_parts(packed_tree, TreeIdx::COUNT * tree_slot_count) };
    let parent_indices = unsafe {
        std::slice::from_raw_parts(node_metadata.add(MetadataIdx::TreeSlot as usize * node_count), node_count)
    };
    let node_valid = unsafe { std::slice::from_raw_parts(node_valid, node_count) };
    let child_ids = unsafe { std::slice::from_raw_parts(child_ids, node_count * expand_width) };
    let child_logprobs = unsafe { std::slice::from_raw_parts(child_logprobs, node_count * expand_width) };
    let child_prune_logprobs = unsafe { std::slice::from_raw_parts(child_prune_logprobs, node_count * expand_width) };
    let frontier = unsafe { std::slice::from_raw_parts_mut(frontier, FrontierIdx::COUNT * frontier_capacity) };

    for index in 0..node_count * expand_width {
        let row = index / expand_width;
        if node_valid[row] == 0 {
            continue;
        }
        let parent = parent_indices[row] as usize;
        let slot = parent * expand_width + index % expand_width;
        if parent >= tree_slot_count || slot >= frontier_capacity {
            continue;
        }
        let cumulative_logprob = path_logprob_bits(
            packed_tree[TreeIdx::PathLogprobBits as usize * tree_slot_count + parent],
            child_logprobs[index].to_bits(),
        );
        frontier[FrontierIdx::TokenId as usize * frontier_capacity + slot] = child_ids[index];
        frontier[FrontierIdx::ParentSlot as usize * frontier_capacity + slot] = parent as u32;
        frontier[FrontierIdx::Depth as usize * frontier_capacity + slot] =
            packed_tree[TreeIdx::Depth as usize * tree_slot_count + parent] + 1;
        frontier[FrontierIdx::PathLogprobBits as usize * frontier_capacity + slot] = cumulative_logprob;
        frontier[FrontierIdx::EdgeLogprobBits as usize * frontier_capacity + slot] =
            child_prune_logprobs[index].to_bits();
        frontier[FrontierIdx::PathScoreKey as usize * frontier_capacity + slot] = top_k_score_key(cumulative_logprob);
        frontier[FrontierIdx::Active as usize * frontier_capacity + slot] = 1;
    }
}

#[cfg(test)]
#[uzu_engine_macros::uzu_test]
fn path_score_bits_are_stable() {
    // Frozen outputs and priorities from the actual CPU insert kernel, before explicit propagation.
    for (parent, child, sum, key) in [
        (0x0000_0000, 0x8000_0000, 0x0000_0000, 0x8000_0000),
        (0x8000_0000, 0x8000_0000, 0x8000_0000, 0x7fff_ffff),
        (0x0000_0001, 0x0000_0001, 0x0000_0002, 0x8000_0002),
        (0x007f_ffff, 0x0000_0001, 0x0080_0000, 0x8080_0000),
        (0x3f80_0000, 0x3380_0000, 0x3f80_0000, 0xbf80_0000),
        (0x3f80_0001, 0x3380_0000, 0x3f80_0002, 0xbf80_0002),
        (0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc0_0000),
        (0xff80_0000, 0x7f80_0000, 0x7fc0_0000, 0xffc0_0000),
        (0x7fc1_2345, 0xffc5_4321, 0xffc5_4321, 0x003a_bcde),
        (0xffc5_4321, 0x7fc1_2345, 0x7fc1_2345, 0xffc1_2345),
        (0x7f81_2345, 0xff85_4321, 0xffc5_4321, 0x003a_bcde),
        (0xff85_4321, 0x7f81_2345, 0x7fc1_2345, 0xffc1_2345),
        (0x7f81_2345, 0xffc5_4321, 0x7fc1_2345, 0xffc1_2345),
        (0x7fc1_2345, 0xff85_4321, 0xffc5_4321, 0x003a_bcde),
        (0x0000_0000, 0xff85_4321, 0xffc5_4321, 0x003a_bcde),
    ] {
        let actual = path_logprob_bits(parent, child);
        assert_eq!(actual, sum, "{parent:#010x} + {child:#010x}");
        assert_eq!(top_k_score_key(actual), key, "priority of {sum:#010x}");
    }
}

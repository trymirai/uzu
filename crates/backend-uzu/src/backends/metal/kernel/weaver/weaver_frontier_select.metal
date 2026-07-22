#include <metal_simdgroup>
#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "weaver_frontier.h"

using namespace metal;

PUBLIC KERNEL(WeaverFrontierSelect)(
    device uint* frontier,
    device uint* tree,
    device uint* slot_ancestors,
    device uint* round_token,
    device uint* round_metadata,
    device uint* round_ancestors,
    device uint* round_valid,
    constant uint& capacity,
    constant uint& tree_slots,
    constant uint& width,
    constant uint& slot_start,
    constant uint& ancestor_stride,
    constant uint& max_depth,
    constant uint& lookahead_count,
    threadgroup uint4 reduce[WEAVER_FRONTIER_SELECT_SIMDGROUPS],
    threadgroup uint winner_slot[WEAVER_FRONTIER_MAX_WIDTH],
    const ThreadContext thread_context,
    const uint group GROUPS(1),
    const uint lid THREADS(256)
) {
  if (capacity == 0 || capacity > WEAVER_FRONTIER_MAX_SLOTS || width == 0 || width > WEAVER_FRONTIER_MAX_WIDTH ||
      ancestor_stride == 0 || max_depth == 0 || tree_slots == 0 || slot_start + width > tree_slots) {
    return;
  }

  bool entry_active[WEAVER_FRONTIER_ENTRIES_PER_THREAD];
  for (uint entry = 0; entry < WEAVER_FRONTIER_ENTRIES_PER_THREAD; ++entry) {
    const uint slot = lid + entry * WEAVER_FRONTIER_SELECT_THREADS;
    entry_active[entry] = slot < capacity && frontier[WEAVER_FRONTIER_LANE_ACTIVE * capacity + slot] != 0u;
  }

  for (uint child = 0; child < width; ++child) {
    uint4 local = uint4(0u, WEAVER_FRONTIER_NO_WINNER, WEAVER_FRONTIER_NO_WINNER, WEAVER_FRONTIER_NO_WINNER);
    for (uint entry = 0; entry < WEAVER_FRONTIER_ENTRIES_PER_THREAD; ++entry) {
      const uint slot = lid + entry * WEAVER_FRONTIER_SELECT_THREADS;
      if (entry_active[entry]) {
        const uint key = frontier[WEAVER_FRONTIER_LANE_KEY * capacity + slot];
        const uint parent = frontier[WEAVER_FRONTIER_LANE_PARENT * capacity + slot];
        const uint token = frontier[WEAVER_FRONTIER_LANE_TOKEN * capacity + slot];
        if (key > local.x || (key == local.x && (parent < local.y || (parent == local.y && token < local.z)))) {
          local = uint4(key, parent, token, slot);
        }
      }
    }

    uint4 simd;
    simd.x = simd_max(local.x);
    simd.y = simd_min(local.x == simd.x ? local.y : WEAVER_FRONTIER_NO_WINNER);
    simd.z = simd_min(all(local.xy == simd.xy) ? local.z : WEAVER_FRONTIER_NO_WINNER);
    simd.w = simd_min(all(local.xyz == simd.xyz) ? local.w : WEAVER_FRONTIER_NO_WINNER);
    if (thread_context.simd_lane_id == 0) {
      reduce[thread_context.simdgroup_index] = simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_context.simdgroup_index == 0) {
      const uint4 partial =
          thread_context.simd_lane_id < WEAVER_FRONTIER_SELECT_SIMDGROUPS
              ? reduce[thread_context.simd_lane_id]
              : uint4(0u, WEAVER_FRONTIER_NO_WINNER, WEAVER_FRONTIER_NO_WINNER, WEAVER_FRONTIER_NO_WINNER);
      uint4 selected;
      selected.x = simd_max(partial.x);
      selected.y = simd_min(partial.x == selected.x ? partial.y : WEAVER_FRONTIER_NO_WINNER);
      selected.z = simd_min(all(partial.xy == selected.xy) ? partial.z : WEAVER_FRONTIER_NO_WINNER);
      selected.w = simd_min(all(partial.xyz == selected.xyz) ? partial.w : WEAVER_FRONTIER_NO_WINNER);
      if (thread_context.simd_lane_id == 0) {
        winner_slot[child] = selected.w;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint selected = winner_slot[child];
    for (uint entry = 0; entry < WEAVER_FRONTIER_ENTRIES_PER_THREAD; ++entry) {
      entry_active[entry] = entry_active[entry] && (lid + entry * WEAVER_FRONTIER_SELECT_THREADS) != selected;
    }
  }

  if (lid >= width) {
    return;
  }

  const uint row = lid;
  const uint slot = winner_slot[row];
  const bool real = slot != WEAVER_FRONTIER_NO_WINNER;
  const uint tree_slot = slot_start + row;

  const uint token = real ? frontier[WEAVER_FRONTIER_LANE_TOKEN * capacity + slot] : 0u;
  const uint parent = real ? frontier[WEAVER_FRONTIER_LANE_PARENT * capacity + slot] : WEAVER_FRONTIER_NO_WINNER;
  const uint depth = real ? frontier[WEAVER_FRONTIER_LANE_DEPTH * capacity + slot] : 0u;
  const uint cum = real ? frontier[WEAVER_FRONTIER_LANE_CUM * capacity + slot] : 0u;
  const uint logprob = real ? frontier[WEAVER_FRONTIER_LANE_LOGPROB * capacity + slot] : 0u;

  tree[WEAVER_TREE_LANE_TOKEN * tree_slots + tree_slot] = token;
  tree[WEAVER_TREE_LANE_PARENT * tree_slots + tree_slot] = parent;
  tree[WEAVER_TREE_LANE_DEPTH * tree_slots + tree_slot] = depth;
  tree[WEAVER_TREE_LANE_CUM * tree_slots + tree_slot] = cum;
  tree[WEAVER_TREE_LANE_LOGPROB * tree_slots + tree_slot] = logprob;
  tree[WEAVER_TREE_LANE_MASK * tree_slots + tree_slot] = real ? 1u : 0u;

  if (real) {
    frontier[WEAVER_FRONTIER_LANE_ACTIVE * capacity + slot] = 0u;
  }

  const uint parent_slot = real && parent < tree_slots ? parent : 0u;
  for (uint index = 0; index < ancestor_stride; ++index) {
    const uint ancestor =
        real && index + 1u <= depth
            ? (index + 1u == depth ? parent_slot : slot_ancestors[parent_slot * ancestor_stride + index])
            : 0u;
    slot_ancestors[tree_slot * ancestor_stride + index] = ancestor;
    round_ancestors[row * ancestor_stride + index] = ancestor;
  }

  round_token[row] = token;
  round_metadata[WEAVER_METADATA_LANE_DEPTH * width + row] = min(depth, max_depth - 1u);
  round_metadata[WEAVER_METADATA_LANE_ANCESTOR_COUNT * width + row] = min(depth, ancestor_stride);
  round_metadata[WEAVER_METADATA_LANE_NODE_INDEX * width + row] = tree_slot;
  round_valid[row] = real && depth < lookahead_count && depth < max_depth ? 1u : 0u;
}

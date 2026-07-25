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
    device uint* round_token_ids,
    device uint* round_metadata,
    device uint* round_ancestors,
    device uint* round_valid,
    const device uint* candidate_pool_ids,
    const device float* candidate_pool_scores,
    device uint* round_candidate_ids,
    device float* round_candidate_scores,
    constant uint& capacity,
    constant uint& tree_slots,
    constant uint& width,
    constant uint& slot_start,
    constant uint& ancestor_stride,
    constant uint& max_depth,
    constant uint& lookahead_count,
    constant uint& candidate_pool_rows,
    constant uint& candidate_pool_size,
    threadgroup uint4 reduce[FRONTIER_SELECT_SIMDGROUPS],
    threadgroup uint winner_slot[FRONTIER_MAX_WIDTH],
    threadgroup uint row_candidate_pool_row[FRONTIER_MAX_WIDTH],
    const ThreadContext thread_context,
    const uint group_index GROUPS(1),
    const uint lid THREADS(FRONTIER_SELECT_THREADS)
) {
  (void)group_index;
  if (capacity == 0 || capacity > FRONTIER_MAX_SLOTS || width == 0 || width > FRONTIER_MAX_WIDTH ||
      ancestor_stride == 0 || max_depth == 0 || tree_slots == 0 || slot_start + width > tree_slots ||
      candidate_pool_rows == 0 || candidate_pool_size == 0) {
    return;
  }

  bool entry_active[FRONTIER_ENTRIES_PER_THREAD];
  for (uint entry = 0; entry < FRONTIER_ENTRIES_PER_THREAD; ++entry) {
    const uint slot = lid + entry * FRONTIER_SELECT_THREADS;
    entry_active[entry] = slot < capacity && frontier[uint(FrontierIdx::Active) * capacity + slot] != 0u;
  }

  for (uint child = 0; child < width; ++child) {
    uint4 local = uint4(0u, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER);
    for (uint entry = 0; entry < FRONTIER_ENTRIES_PER_THREAD; ++entry) {
      const uint slot = lid + entry * FRONTIER_SELECT_THREADS;
      if (entry_active[entry]) {
        const uint key = frontier[uint(FrontierIdx::PathScoreKey) * capacity + slot];
        const uint parent = frontier[uint(FrontierIdx::ParentSlot) * capacity + slot];
        const uint token = frontier[uint(FrontierIdx::TokenId) * capacity + slot];
        if (key > local.x || (key == local.x && (parent < local.y || (parent == local.y && token < local.z)))) {
          local = uint4(key, parent, token, slot);
        }
      }
    }

    uint4 simd;
    simd.x = simd_max(local.x);
    simd.y = simd_min(local.x == simd.x ? local.y : FRONTIER_NO_WINNER);
    simd.z = simd_min(all(local.xy == simd.xy) ? local.z : FRONTIER_NO_WINNER);
    simd.w = simd_min(all(local.xyz == simd.xyz) ? local.w : FRONTIER_NO_WINNER);
    if (thread_context.simd_lane_id == 0) {
      reduce[thread_context.simdgroup_index] = simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_context.simdgroup_index == 0) {
      const uint4 partial = thread_context.simd_lane_id < FRONTIER_SELECT_SIMDGROUPS
                                ? reduce[thread_context.simd_lane_id]
                                : uint4(0u, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER);
      uint4 selected;
      selected.x = simd_max(partial.x);
      selected.y = simd_min(partial.x == selected.x ? partial.y : FRONTIER_NO_WINNER);
      selected.z = simd_min(all(partial.xy == selected.xy) ? partial.z : FRONTIER_NO_WINNER);
      selected.w = simd_min(all(partial.xyz == selected.xyz) ? partial.w : FRONTIER_NO_WINNER);
      if (thread_context.simd_lane_id == 0) {
        winner_slot[child] = selected.w;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint selected = winner_slot[child];
    for (uint entry = 0; entry < FRONTIER_ENTRIES_PER_THREAD; ++entry) {
      entry_active[entry] = entry_active[entry] && (lid + entry * FRONTIER_SELECT_THREADS) != selected;
    }
  }

  if (lid < width) {
    const uint row = lid;
    const uint slot = winner_slot[row];
    const bool real = slot != FRONTIER_NO_WINNER;
    const uint tree_slot = slot_start + row;

    const uint token = real ? frontier[uint(FrontierIdx::TokenId) * capacity + slot] : 0u;
    const uint parent = real ? frontier[uint(FrontierIdx::ParentSlot) * capacity + slot] : FRONTIER_NO_WINNER;
    const uint depth = real ? frontier[uint(FrontierIdx::Depth) * capacity + slot] : 0u;
    const uint cumulative_logprob = real ? frontier[uint(FrontierIdx::PathLogprobBits) * capacity + slot] : 0u;
    const uint logprob = real ? frontier[uint(FrontierIdx::EdgeLogprobBits) * capacity + slot] : 0u;

    tree[uint(TreeIdx::TokenId) * tree_slots + tree_slot] = token;
    tree[uint(TreeIdx::ParentSlot) * tree_slots + tree_slot] = parent;
    tree[uint(TreeIdx::Depth) * tree_slots + tree_slot] = depth;
    tree[uint(TreeIdx::PathLogprobBits) * tree_slots + tree_slot] = cumulative_logprob;
    tree[uint(TreeIdx::EdgeLogprobBits) * tree_slots + tree_slot] = logprob;
    tree[uint(TreeIdx::Valid) * tree_slots + tree_slot] = real ? 1u : 0u;

    if (real) {
      frontier[uint(FrontierIdx::Active) * capacity + slot] = 0u;
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

    round_token_ids[row] = token;
    round_metadata[uint(MetadataIdx::Depth) * width + row] = min(depth, max_depth - 1u);
    round_metadata[uint(MetadataIdx::AncestorCount) * width + row] = min(depth, ancestor_stride);
    round_metadata[uint(MetadataIdx::TreeSlot) * width + row] = tree_slot;
    round_valid[row] = real && depth < lookahead_count && depth < max_depth ? 1u : 0u;

    row_candidate_pool_row[row] = min(depth, candidate_pool_rows - 1u);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint row = 0; row < width; ++row) {
    const uint source = row_candidate_pool_row[row] * candidate_pool_size;
    const uint destination = row * candidate_pool_size;
    for (uint candidate = lid; candidate < candidate_pool_size; candidate += FRONTIER_SELECT_THREADS) {
      round_candidate_ids[destination + candidate] = candidate_pool_ids[source + candidate];
      round_candidate_scores[destination + candidate] = candidate_pool_scores[source + candidate];
    }
  }
}

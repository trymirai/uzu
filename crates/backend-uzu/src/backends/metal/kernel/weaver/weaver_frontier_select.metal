#include <metal_simdgroup>
#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "weaver_frontier.h"

using namespace metal;

PUBLIC KERNEL(WeaverFrontierSelect)(
    device uint* frontier,
    device uint* packed_tree,
    device uint* slot_ancestors,
    device uint* node_token_ids,
    device uint* node_metadata,
    device uint* node_ancestor_indices,
    device uint* node_valid,
    const device uint* candidate_pool_ids,
    const device float* candidate_pool_logits,
    device uint* node_candidate_ids,
    device float* node_candidate_logits,
    constant uint& frontier_capacity,
    constant uint& tree_slot_count,
    constant uint& node_count,
    constant uint& batch_start_slot,
    constant uint& ancestor_stride,
    constant uint& max_depth,
    constant uint& lookahead_count,
    constant uint& candidate_pool_row_count,
    constant uint& candidates_per_row,
    threadgroup uint4 reduce[FRONTIER_SELECT_SIMDGROUPS],
    threadgroup uint winner_slot[FRONTIER_MAX_WIDTH],
    threadgroup uint node_candidate_pool_row[FRONTIER_MAX_WIDTH],
    const ThreadContext thread_context,
    const uint group_index GROUPS(1),
    const uint lid THREADS(FRONTIER_SELECT_THREADS)
) {
  (void)group_index;
  if (frontier_capacity == 0 || frontier_capacity > FRONTIER_MAX_SLOTS || node_count == 0 ||
      node_count > FRONTIER_MAX_WIDTH || ancestor_stride == 0 || max_depth == 0 || tree_slot_count == 0 ||
      batch_start_slot + node_count > tree_slot_count || candidate_pool_row_count == 0 || candidates_per_row == 0) {
    return;
  }

  bool entry_active[FRONTIER_ENTRIES_PER_THREAD];
  for (uint entry = 0; entry < FRONTIER_ENTRIES_PER_THREAD; ++entry) {
    const uint slot = lid + entry * FRONTIER_SELECT_THREADS;
    entry_active[entry] =
        slot < frontier_capacity && frontier[uint(FrontierIdx::Active) * frontier_capacity + slot] != 0u;
  }

  for (uint child = 0; child < node_count; ++child) {
    uint4 local = uint4(0u, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER);
    for (uint entry = 0; entry < FRONTIER_ENTRIES_PER_THREAD; ++entry) {
      const uint slot = lid + entry * FRONTIER_SELECT_THREADS;
      if (entry_active[entry]) {
        const uint key = frontier[uint(FrontierIdx::PathScoreKey) * frontier_capacity + slot];
        const uint parent = frontier[uint(FrontierIdx::ParentSlot) * frontier_capacity + slot];
        const uint token = frontier[uint(FrontierIdx::TokenId) * frontier_capacity + slot];
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

  if (lid < node_count) {
    const uint row = lid;
    const uint slot = winner_slot[row];
    const bool real = slot != FRONTIER_NO_WINNER;
    const uint tree_slot = batch_start_slot + row;

    const uint token = real ? frontier[uint(FrontierIdx::TokenId) * frontier_capacity + slot] : 0u;
    const uint parent = real ? frontier[uint(FrontierIdx::ParentSlot) * frontier_capacity + slot] : FRONTIER_NO_WINNER;
    const uint depth = real ? frontier[uint(FrontierIdx::Depth) * frontier_capacity + slot] : 0u;
    const uint cumulative_logprob = real ? frontier[uint(FrontierIdx::PathLogprobBits) * frontier_capacity + slot] : 0u;
    const uint logprob = real ? frontier[uint(FrontierIdx::EdgeLogprobBits) * frontier_capacity + slot] : 0u;

    packed_tree[uint(TreeIdx::TokenId) * tree_slot_count + tree_slot] = token;
    packed_tree[uint(TreeIdx::ParentSlot) * tree_slot_count + tree_slot] = parent;
    packed_tree[uint(TreeIdx::Depth) * tree_slot_count + tree_slot] = depth;
    packed_tree[uint(TreeIdx::PathLogprobBits) * tree_slot_count + tree_slot] = cumulative_logprob;
    packed_tree[uint(TreeIdx::EdgeLogprobBits) * tree_slot_count + tree_slot] = logprob;
    packed_tree[uint(TreeIdx::Valid) * tree_slot_count + tree_slot] = real ? 1u : 0u;

    if (real) {
      frontier[uint(FrontierIdx::Active) * frontier_capacity + slot] = 0u;
    }

    const uint parent_slot = real && parent < tree_slot_count ? parent : 0u;
    for (uint index = 0; index < ancestor_stride; ++index) {
      const uint ancestor =
          real && index + 1u <= depth
              ? (index + 1u == depth ? parent_slot : slot_ancestors[parent_slot * ancestor_stride + index])
              : 0u;
      slot_ancestors[tree_slot * ancestor_stride + index] = ancestor;
      node_ancestor_indices[row * ancestor_stride + index] = ancestor;
    }

    node_token_ids[row] = token;
    node_metadata[uint(MetadataIdx::Depth) * node_count + row] = min(depth, max_depth - 1u);
    node_metadata[uint(MetadataIdx::AncestorCount) * node_count + row] = min(depth, ancestor_stride);
    node_metadata[uint(MetadataIdx::TreeSlot) * node_count + row] = tree_slot;
    node_valid[row] = real && depth < lookahead_count && depth < max_depth ? 1u : 0u;

    node_candidate_pool_row[row] = min(depth, candidate_pool_row_count - 1u);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint row = 0; row < node_count; ++row) {
    const uint source = node_candidate_pool_row[row] * candidates_per_row;
    const uint destination = row * candidates_per_row;
    for (uint candidate = lid; candidate < candidates_per_row; candidate += FRONTIER_SELECT_THREADS) {
      node_candidate_ids[destination + candidate] = candidate_pool_ids[source + candidate];
      node_candidate_logits[destination + candidate] = candidate_pool_logits[source + candidate];
    }
  }
}

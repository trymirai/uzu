#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/top_k.h"
#include "weaver_frontier.h"

using namespace metal;

PUBLIC KERNEL(WeaverFrontierInsertChildren)(
    const device uint* packed_tree,
    const device uint* node_metadata,
    const device uint* node_valid,
    const device uint* child_ids,
    const device float* child_logprobs,
    device uint* frontier,
    constant uint& frontier_capacity,
    constant uint& tree_slot_count,
    constant uint& node_count,
    constant uint& expand_width,
    const uint position AXIS(node_count* expand_width, 64)
) {
  if (frontier_capacity == 0 || tree_slot_count == 0 || expand_width == 0) {
    return;
  }

  const uint row = position / expand_width, child = position % expand_width;
  if (node_valid[row] == 0u) {
    return;
  }

  const uint parent = node_metadata[uint(MetadataIdx::TreeSlot) * node_count + row];
  if (parent >= tree_slot_count) {
    return;
  }
  const uint slot = parent * expand_width + child;
  if (slot >= frontier_capacity) {
    return;
  }

  const float logprob = child_logprobs[row * expand_width + child];
  const float cumulative_logprob =
      as_type<float>(packed_tree[uint(TreeIdx::PathLogprobBits) * tree_slot_count + parent]) + logprob;

  frontier[uint(FrontierIdx::TokenId) * frontier_capacity + slot] = child_ids[row * expand_width + child];
  frontier[uint(FrontierIdx::ParentSlot) * frontier_capacity + slot] = parent;
  frontier[uint(FrontierIdx::Depth) * frontier_capacity + slot] =
      packed_tree[uint(TreeIdx::Depth) * tree_slot_count + parent] + 1u;
  frontier[uint(FrontierIdx::PathLogprobBits) * frontier_capacity + slot] = as_type<uint>(cumulative_logprob);
  frontier[uint(FrontierIdx::EdgeLogprobBits) * frontier_capacity + slot] = as_type<uint>(logprob);
  frontier[uint(FrontierIdx::PathScoreKey) * frontier_capacity + slot] = top_k_score_key(cumulative_logprob);
  frontier[uint(FrontierIdx::Active) * frontier_capacity + slot] = 1u;
}

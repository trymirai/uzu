#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/top_k.h"
#include "weaver_frontier.h"

using namespace metal;

PUBLIC KERNEL(WeaverFrontierScatter)(
    const device uint* packed_tree,
    const device uint* node_metadata,
    const device uint* node_valid,
    const device uint* child_ids,
    const device float* child_model_logprobs,
    device uint* frontier,
    constant uint& frontier_capacity,
    constant uint& tree_slot_count,
    constant uint& node_count,
    constant uint& children_per_node,
    const uint position AXIS(node_count* children_per_node, 64)
) {
  if (frontier_capacity == 0 || tree_slot_count == 0 || children_per_node == 0) {
    return;
  }

  const uint row = position / children_per_node, child = position % children_per_node;
  if (node_valid[row] == 0u) {
    return;
  }

  const uint parent = node_metadata[uint(MetadataIdx::TreeSlot) * node_count + row];
  if (parent >= tree_slot_count) {
    return;
  }
  const uint slot = parent * children_per_node + child;
  if (slot >= frontier_capacity) {
    return;
  }

  const float logprob = child_model_logprobs[row * children_per_node + child];
  const float cumulative_logprob =
      as_type<float>(packed_tree[uint(TreeIdx::PathLogprobBits) * tree_slot_count + parent]) + logprob;

  frontier[uint(FrontierIdx::TokenId) * frontier_capacity + slot] = child_ids[row * children_per_node + child];
  frontier[uint(FrontierIdx::ParentSlot) * frontier_capacity + slot] = parent;
  frontier[uint(FrontierIdx::Depth) * frontier_capacity + slot] =
      packed_tree[uint(TreeIdx::Depth) * tree_slot_count + parent] + 1u;
  frontier[uint(FrontierIdx::PathLogprobBits) * frontier_capacity + slot] = as_type<uint>(cumulative_logprob);
  frontier[uint(FrontierIdx::EdgeLogprobBits) * frontier_capacity + slot] = as_type<uint>(logprob);
  frontier[uint(FrontierIdx::PathScoreKey) * frontier_capacity + slot] = top_k_score_key(cumulative_logprob);
  frontier[uint(FrontierIdx::Active) * frontier_capacity + slot] = 1u;
}

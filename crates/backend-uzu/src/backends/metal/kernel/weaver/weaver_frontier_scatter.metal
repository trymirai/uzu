#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/top_k.h"
#include "weaver_frontier.h"

using namespace metal;

PUBLIC KERNEL(WeaverFrontierScatter)(
    const device uint* tree,
    const device uint* round_metadata,
    const device uint* round_valid,
    const device uint* child_ids,
    const device float* child_logprobs,
    device uint* frontier,
    constant uint& capacity,
    constant uint& tree_slots,
    constant uint& rows,
    constant uint& children_per_node,
    const uint position AXIS(rows* children_per_node, 64)
) {
  if (capacity == 0 || tree_slots == 0 || children_per_node == 0) {
    return;
  }

  const uint row = position / children_per_node, child = position % children_per_node;
  if (round_valid[row] == 0u) {
    return;
  }

  const uint parent = round_metadata[uint(MetadataIdx::TreeSlot) * rows + row];
  if (parent >= tree_slots) {
    return;
  }
  const uint slot = parent * children_per_node + child;
  if (slot >= capacity) {
    return;
  }

  const float logprob = child_logprobs[row * children_per_node + child];
  const float cumulative_logprob = as_type<float>(tree[uint(TreeIdx::PathLogprobBits) * tree_slots + parent]) + logprob;

  frontier[uint(FrontierIdx::TokenId) * capacity + slot] = child_ids[row * children_per_node + child];
  frontier[uint(FrontierIdx::ParentSlot) * capacity + slot] = parent;
  frontier[uint(FrontierIdx::Depth) * capacity + slot] = tree[uint(TreeIdx::Depth) * tree_slots + parent] + 1u;
  frontier[uint(FrontierIdx::PathLogprobBits) * capacity + slot] = as_type<uint>(cumulative_logprob);
  frontier[uint(FrontierIdx::EdgeLogprobBits) * capacity + slot] = as_type<uint>(logprob);
  frontier[uint(FrontierIdx::PathScoreKey) * capacity + slot] = top_k_score_key(cumulative_logprob);
  frontier[uint(FrontierIdx::Active) * capacity + slot] = 1u;
}

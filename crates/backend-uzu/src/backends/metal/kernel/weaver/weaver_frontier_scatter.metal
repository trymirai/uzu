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
    constant uint& fanout,
    const uint position AXIS(rows* fanout, 64)
) {
  if (capacity == 0 || tree_slots == 0 || fanout == 0) {
    return;
  }

  const uint row = position / fanout, child = position % fanout;
  if (round_valid[row] == 0u) {
    return;
  }

  const uint parent = round_metadata[WEAVER_METADATA_LANE_NODE_INDEX * rows + row];
  if (parent >= tree_slots) {
    return;
  }
  const uint slot = parent * fanout + child;
  if (slot >= capacity) {
    return;
  }

  const float logprob = child_logprobs[row * fanout + child];
  const float cumulative_logprob = as_type<float>(tree[WEAVER_TREE_LANE_CUM * tree_slots + parent]) + logprob;

  frontier[WEAVER_FRONTIER_LANE_TOKEN * capacity + slot] = child_ids[row * fanout + child];
  frontier[WEAVER_FRONTIER_LANE_PARENT * capacity + slot] = parent;
  frontier[WEAVER_FRONTIER_LANE_DEPTH * capacity + slot] = tree[WEAVER_TREE_LANE_DEPTH * tree_slots + parent] + 1u;
  frontier[WEAVER_FRONTIER_LANE_CUM * capacity + slot] = as_type<uint>(cumulative_logprob);
  frontier[WEAVER_FRONTIER_LANE_LOGPROB * capacity + slot] = as_type<uint>(logprob);
  frontier[WEAVER_FRONTIER_LANE_KEY * capacity + slot] = top_k_score_key(cumulative_logprob);
  frontier[WEAVER_FRONTIER_LANE_ACTIVE * capacity + slot] = 1u;
}

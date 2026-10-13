#include <metal_stdlib>
#include "../common/dsl.h"
#include "../weaver/weaver_frontier.h"
#include "confidence_head.h"

using namespace metal;

// The full head's edge of every real child of a slot (one threadgroup per slot): z_u and the children's token vectors
// corrected by the tree attention, alpha, beta and delta as in the local part, log q over the slot's kept pool
// logits and the pool row's noise. min(log q, 0) goes to the edge lane the final prune sums along paths: the packed
// tree's for a child expanded into a slot, the frontier's otherwise. The CPU twin is the reference.
PUBLIC KERNEL(ConfidenceHeadPrune)(
    const device float* nodes,
    const device float* child_inputs,
    const device uint* child_indices,
    const device uint* child_real,
    const device float* slot_logits,
    const device float* noise,
    const device float* node_shifts,
    const device float* child_shifts,
    const device uint* slot_tokens,
    const device uint* token_slots,
    const device float* local_weights,
    device uint* packed_tree,
    device uint* frontier,
    constant uint& slot_count,
    constant uint& pool_rows,
    constant float& cap,
    constant float& epsilon,
    const uint candidates SPECIALIZE,
    const uint children SPECIALIZE,
    const uint hidden SPECIALIZE,
    const uint pool_hidden SPECIALIZE,
    const uint edge_hidden SPECIALIZE,
    threadgroup float shared[2 * CH_PRUNE_SIMDGROUPS],
    threadgroup float node[CH_HIDDEN_MAX],
    threadgroup float vectors[CH_CHILDREN_MAX * CH_HIDDEN_MAX],
    threadgroup float numbers[CH_CHILDREN_MAX * CH_SLOT],
    threadgroup uint indices[CH_CHILDREN_MAX],
    threadgroup uint real[CH_CHILDREN_MAX],
    threadgroup float edge_in[CH_CHILDREN_MAX * CH_EDGE_INPUT_MAX],
    threadgroup float edge_state[CH_CHILDREN_MAX * CH_HIDDEN_MAX],
    threadgroup float edge_raised[CH_CHILDREN_MAX * CH_EDGE_HIDDEN_MAX],
    threadgroup float deltas[CH_CHILDREN_MAX],
    threadgroup float edges[CH_CHILDREN_MAX],
    threadgroup float scalars[2],
    const ThreadContext thread_context,
    const uint slot GROUPS(slot_count),
    const uint lid THREADS(CH_PRUNE_THREADS)
) {
  bool any_real = false;
  for (uint child = 0; child < children; ++child) {
    any_real = any_real || child_real[slot * children + child] != 0u;
  }
  if (!any_real) {
    return;
  }
  const ConfidenceHeadLayout layout = confidence_head_layout(hidden, pool_hidden, edge_hidden);
  const device float* weights = local_weights;
  const uint depth = packed_tree[uint(TreeIdx::Depth) * slot_count + slot];
  const uint pool_row = min(depth, pool_rows - 1u);
  const uint token = slot_tokens[slot];
  const uint input_width = hidden + CH_SLOT;

  if (lid < hidden) {
    node[lid] = nodes[slot * hidden + lid] + node_shifts[token * hidden + lid];
  }
  for (uint position = lid; position < children * hidden; position += CH_PRUNE_THREADS) {
    const uint child = position / hidden, unit = position % hidden;
    const uint kept = slot * children + child;
    vectors[position] = child_inputs[kept * input_width + unit] + child_shifts[(1u + kept) * hidden + unit];
  }
  if (lid < children * CH_SLOT) {
    const uint child = lid / CH_SLOT, number = lid % CH_SLOT;
    numbers[lid] = child_inputs[(slot * children + child) * input_width + hidden + number];
  }
  if (lid < children) {
    indices[lid] = child_indices[slot * children + lid];
    real[lid] = child_real[slot * children + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  confidence_head_alpha_beta<CH_PRUNE_THREADS>(node, weights, layout, hidden, cap, scalars, thread_context, lid);

  // This thread's candidate.
  const uint candidate = lid;
  const bool valid = candidate < candidates && candidate < CH_CANDIDATES_MAX;
  const float logit = valid ? slot_logits[slot * candidates + candidate] : -INFINITY;
  const float gumbel = valid ? noise[pool_row * candidates + candidate] : 0.0f;
  confidence_head_edge_deltas<CH_PRUNE_THREADS>(
      node,
      vectors,
      numbers,
      edge_in,
      edge_state,
      edge_raised,
      deltas,
      weights,
      layout,
      children,
      hidden,
      edge_hidden,
      epsilon,
      thread_context,
      lid
  );
  confidence_head_child_edges<CH_PRUNE_THREADS>(
      logit, gumbel, valid, candidate, indices, deltas, real, edges, shared, scalars, children, thread_context
  );
  if (lid < children && real[lid] != 0u) {
    const uint kept = slot * children + lid;
    const uint expanded = token_slots[1u + kept];
    const uint bits = as_type<uint>(edges[lid]);
    if (expanded != CH_NONE) {
      packed_tree[uint(TreeIdx::EdgeLogprobBits) * slot_count + expanded] = bits;
    } else {
      frontier[uint(FrontierIdx::EdgeLogprobBits) * slot_count * children + kept] = bits;
    }
  }
}

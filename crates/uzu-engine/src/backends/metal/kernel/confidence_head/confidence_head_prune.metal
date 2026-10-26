#include <metal_stdlib>
#include "../common/dsl.h"
#include "../weaver/weaver_frontier.h"
#include "confidence_head_common.h"

using namespace metal;

// One threadgroup per tree slot, with one thread per candidate during edge normalization.
#define PRUNE_THREADS 512u
#define PRUNE_SIMDGROUPS (PRUNE_THREADS / METAL_SIMD_SIZE)

// Refine saved node and child features with tree-attention shifts, then recompute edge scores.
// Reuse the local head's alpha/beta and edge MLP with the original pool logits and noise.
// Only edge scores are updated; the subsequent budget prune accumulates them along paths.
// This kernel does not remove nodes or update the frontier's path scores and priority keys.
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
    const device float* weights,
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
    threadgroup float shared[2 * PRUNE_SIMDGROUPS],
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
    const uint lid THREADS(PRUNE_THREADS)
) {
  bool any_real = false;
  for (uint child = 0; child < children; ++child) {
    any_real = any_real || child_real[slot * children + child] != 0u;
  }
  if (!any_real) {
    return;
  }

  // 1. Locate the packed weights and select the saved noise row for this parent's depth.
  // Depths beyond the prepared pool reuse its last row.
  const ConfidenceHeadLayout layout = confidence_head_layout(hidden, pool_hidden, edge_hidden);
  const uint depth = packed_tree[uint(TreeIdx::Depth) * slot_count + slot];
  const uint pool_row = min(depth, pool_rows - 1u);
  const uint token = slot_tokens[slot];
  const uint input_width = hidden + CH_SLOT;

  // 2. Prepare refined node and child features in threadgroup memory.
  // Refine the parent vector: z' = z + node_shift. The saved vector is indexed by
  // tree slot, while the shift is indexed by tree-attention position.
  if (lid < hidden) {
    node[lid] = nodes[slot * hidden + lid] + node_shifts[token * hidden + lid];
  }
  // Refine each child token vector: e' = e + child_shift. Tree-attention position 0
  // is reserved for the root; child record kept occupies position 1 + kept.
  for (uint position = lid; position < children * hidden; position += PRUNE_THREADS) {
    const uint child = position / hidden;
    const uint unit = position % hidden;
    const uint kept = slot * children + child;
    vectors[position] = child_inputs[kept * input_width + unit] + child_shifts[(1u + kept) * hidden + unit];
  }
  // Reuse the four standardized scalar features from encode_round without changes:
  // draft log probability, corrected log probability, Gumbel noise and log(1 + draft rank).
  if (lid < children * CH_SLOT) {
    const uint child = lid / CH_SLOT;
    const uint number = lid % CH_SLOT;
    numbers[lid] = child_inputs[(slot * children + child) * input_width + hidden + number];
  }
  // Load each child's original pool index and presence flag. Publish the prepared
  // node, vectors, numbers and metadata before the shared scoring helpers read them.
  if (lid < children) {
    indices[lid] = child_indices[slot * children + lid];
    real[lid] = child_real[slot * children + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. Recompute alpha, beta and child corrections using the refined features.
  // Recompute capped softplus coefficients alpha and beta from the refined parent
  // vector. The helper stores them in scalars and ends with a threadgroup barrier.
  confidence_head_alpha_beta<PRUNE_THREADS>(node, weights, layout, hidden, cap, scalars, thread_context, lid);
  // Assign one candidate to each thread and reuse its saved Weaver-corrected logit
  // and target Gumbel noise. Threads beyond the pool are excluded from normalization.
  const uint candidate = lid;
  const bool valid = candidate < candidates && candidate < CANDIDATES_MAX;
  const float logit = valid ? slot_logits[slot * candidates + candidate] : -INFINITY;
  const float gumbel = valid ? noise[pool_row * candidates + candidate] : 0.0f;
  // Recompute child corrections delta with the same edge MLP as encode_round,
  // using [z'; e'; elementwise z' * e'; saved scalar features].
  confidence_head_edge_deltas<PRUNE_THREADS>(
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

  // 4. Compute normalized edge scores over the entire candidate pool.
  // Score the whole pool as alpha * logit + beta * Gumbel, adding delta only at
  // real children's indices. Normalize over all candidates and keep min(log q(child), 0).
  confidence_head_child_edges<PRUNE_THREADS>(
      logit,
      gumbel,
      valid,
      candidate,
      indices,
      deltas,
      real,
      edges,
      shared,
      scalars,
      children,
      thread_context
  );

  // 5. Store the updated edge scores in the tree or frontier.
  // Write each real child's incoming edge score to its current location. An expanded
  // child has its own packed-tree slot; an unexpanded child remains in the frontier.
  // Preserve the float bits in the uint buffer. Cumulative path scores are not rewritten.
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

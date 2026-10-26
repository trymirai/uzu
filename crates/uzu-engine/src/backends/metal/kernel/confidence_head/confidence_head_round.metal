#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/top_k.h"
#include "../weaver/weaver_frontier.h"
#include "confidence_head_common.h"

using namespace metal;

// One round threadgroup processes one node with 1024 threads (32 SIMD groups).
// Each half covers the same 512 candidate indices. Both compute the full pool entry layer;
// the second layer's output channels are split into fixed ranges 0..7 and 8..15.
#define LOCAL_THREADS 1024u
#define LOCAL_SIMDGROUPS (LOCAL_THREADS / METAL_SIMD_SIZE)

// Copies of the pool in the local threadgroup, and the pool encoder's hidden units each copy computes.
#define POOL_COPIES (LOCAL_THREADS / CANDIDATES_MAX)
#define POOL_HALF (POOL_HIDDEN_MAX / POOL_COPIES)
#define POOL_HIDDEN_MAX 16u
#define POOL_WEIGHTS_MAX (POOL_HIDDEN_MAX * (CH_POOL_INPUTS + 2u + POOL_HIDDEN_MAX))
#define LOG_PROB_FLOOR (-30.0f)

template <uint SIMDGROUPS>
METAL_FUNC float4 confidence_head_log_sum_exp2(
    float2 value,
    bool counted,
    threadgroup float4* shared,
    const thread ThreadContext& context
) {
  static_assert(SIMDGROUPS <= METAL_SIMD_SIZE, "a lane per simdgroup");
  const uint lane = context.simd_lane_id;
  const float2 top = simd_max(value);
  const float2 total = simd_sum(counted ? exp(value - top) : float2(0.0f));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    shared[context.simdgroup_index] = float4(top, total);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float4 group = lane < SIMDGROUPS ? shared[lane] : float4(-INFINITY, -INFINITY, 0.0f, 0.0f);
  const float2 maximum = simd_max(group.xy);
  const float2 scaled = select(float2(0.0f), group.zw * exp(group.xy - maximum), group.zw > 0.0f);
  return float4(maximum, simd_sum(scaled));
}

static_assert(POOL_COPIES * CANDIDATES_MAX == LOCAL_THREADS, "every copy of the pool has a thread per candidate");

// Rescore the children already selected by Weaver, one threadgroup per expanded node.
// Write min(log q(child), 0), the cumulative path score and its sorting key to the frontier.
// Save node and child features for final tree refinement. Child token IDs remain unchanged.
// Inputs require a nonempty pool of distinct token IDs containing every selected child.
PUBLIC KERNEL(ConfidenceHeadRound)(
    const device bfloat* residual_logits,
    const device float* candidate_logits,
    const device uint* candidate_ids,
    const device float* noise,
    const device float* draft_rows,
    const device uint* node_metadata,
    const device uint* node_valid,
    const device uint* packed_tree,
    const device float* token_table,
    const device float* weights,
    device float* child_edges_out,
    device uint* frontier,
    device float* nodes,
    device float* child_inputs,
    device uint* child_indices,
    device uint* child_real,
    device float* slot_logits,
    constant uint& rows,
    constant uint& pool_rows,
    constant float& cap,
    constant float& epsilon,
    constant uint& frontier_capacity,
    constant uint& tree_slot_count,
    const uint candidates SPECIALIZE,
    const uint children SPECIALIZE,
    const uint hidden SPECIALIZE,
    const uint pool_hidden SPECIALIZE,
    const uint edge_hidden SPECIALIZE,
    threadgroup float shared[2 * LOCAL_SIMDGROUPS],
    threadgroup float4 shared4[LOCAL_SIMDGROUPS],
    threadgroup float pool_weights[POOL_WEIGHTS_MAX],
    threadgroup float4 pool_hidden_rows[POOL_HIDDEN_MAX * POOL_HIDDEN_MAX / 4],
    threadgroup float partial_mean[LOCAL_SIMDGROUPS * POOL_HALF],
    threadgroup float partial_peak[LOCAL_SIMDGROUPS * POOL_HALF],
    threadgroup float summary[2 * POOL_HIDDEN_MAX],
    threadgroup float node[CH_HIDDEN_MAX],
    threadgroup float vectors[CH_CHILDREN_MAX * CH_HIDDEN_MAX],
    threadgroup float numbers[CH_CHILDREN_MAX * CH_SLOT],
    threadgroup uint child_tokens[CH_CHILDREN_MAX],
    threadgroup uint indices[CH_CHILDREN_MAX],
    threadgroup uint real[CH_CHILDREN_MAX],
    threadgroup float edge_in[CH_CHILDREN_MAX * CH_EDGE_INPUT_MAX],
    threadgroup float edge_state[CH_CHILDREN_MAX * CH_HIDDEN_MAX],
    threadgroup float edge_raised[CH_CHILDREN_MAX * CH_EDGE_HIDDEN_MAX],
    threadgroup float deltas[CH_CHILDREN_MAX],
    threadgroup float edges[CH_CHILDREN_MAX],
    threadgroup float scalars[2],
    const ThreadContext thread_context,
    const uint row GROUPS(rows),
    const uint lid THREADS(LOCAL_THREADS)
) {
  // 1. Skip unused node rows uniformly across the threadgroup, before any barriers.
  if (node_valid[row] == 0u) {
    return;
  }

  // 2. Locate the node in the tree and select its per-depth draft/noise row.
  // Depths beyond the prepared pool reuse the last row.
  const ConfidenceHeadLayout layout = confidence_head_layout(hidden, pool_hidden, edge_hidden);
  const uint depth = node_metadata[uint(MetadataIdx::Depth) * rows + row];
  const uint slot = node_metadata[uint(MetadataIdx::TreeSlot) * rows + row];
  const uint pool_row = min(depth, pool_rows - 1u);
  const uint base = row * candidates;
  const device uint* child_token_ids = frontier + uint(FrontierIdx::TokenId) * frontier_capacity + slot * children;

  // 3. Cache shared pool-encoder weights. Pad matrix rows for float4 dot products.
  // Copy the contiguous [entry matrix, entry bias, hidden matrix, hidden bias] block.
  // The hidden matrix is also staged separately below; pool_weights supplies its bias.
  // Barriers inside confidence_head_log_sum_exp2 publish these writes before the encoder reads them.
  const uint pool_weight_count = pool_hidden * (CH_POOL_INPUTS + 1u + pool_hidden + 1u);
  for (uint index = lid; index < pool_weight_count; index += LOCAL_THREADS) {
    pool_weights[index] = weights[layout.pool_entry_weights + index];
  }
  // The hidden layer's rows again, padded with zeros to POOL_HIDDEN_MAX columns: read as float4.
  for (uint index = lid; index < POOL_HIDDEN_MAX * POOL_HIDDEN_MAX; index += LOCAL_THREADS) {
    const uint unit = index / POOL_HIDDEN_MAX;
    const uint inner = index % POOL_HIDDEN_MAX;
    const float value = unit < pool_hidden && inner < pool_hidden
                            ? weights[layout.pool_hidden_weights + unit * pool_hidden + inner]
                            : 0.0f;
    reinterpret_cast<threadgroup float*>(pool_hidden_rows)[index] = value;
  }

  // 4. Load the children already selected by Weaver and their projected token vectors.
  // This kernel updates their scores without changing the selected token IDs.
  // These loads do not depend on the softmax. Its internal threadgroup barriers publish them
  // before other threads read child_tokens or vectors. indices starts at zero until token matching.
  if (lid < children) {
    child_tokens[lid] = child_token_ids[lid];
    indices[lid] = 0u;
    real[lid] = 1u;
  }
  for (uint position = lid; position < children * hidden; position += LOCAL_THREADS) {
    const uint child = position / hidden, unit = position % hidden;
    vectors[position] = float(token_table[child_token_ids[child] * hidden + unit]);
  }

  // 5. Map threads to two copies of the candidate pool. owner selects the first copy
  // for softmax sums, rank counts and unique writes. Both copies participate in softmax
  // maxima (duplicates do not change a max) and in pooling their assigned feature channels.
  const uint candidate = lid % CANDIDATES_MAX;
  const uint half_index = lid / CANDIDATES_MAX;
  const bool valid = candidate < candidates;
  const bool owner = valid && half_index == 0u;
  const float draft = valid ? candidate_logits[base + candidate] : -INFINITY;
  const float logit = valid ? draft + float(residual_logits[base + candidate]) : -INFINITY;
  const float gumbel = valid ? noise[pool_row * candidates + candidate] : 0.0f;
  const uint token = valid ? candidate_ids[base + candidate] : CH_NONE;
  if (owner) {
    slot_logits[slot * candidates + candidate] = logit;
  }

  // 6. Normalize draft and corrected logits over the entire pool.
  // Clamp log-probability features to -30, but keep the softmax weights used for averaging unclamped.
  // log pi_a, log pi_l (floored) and pi_l over the pool: both log-sum-exps in one exchange.
  const float4 normalisers =
      confidence_head_log_sum_exp2<LOCAL_SIMDGROUPS>(float2(draft, logit), owner, shared4, thread_context);
  const float draft_max = normalisers.x, logit_max = normalisers.y;
  const float2 totals = normalisers.zw;
  const float log_draft = max((draft - draft_max) - log(totals.x), LOG_PROB_FLOOR);
  const float log_logit = max((logit - logit_max) - log(totals.y), LOG_PROB_FLOOR);
  const float weight = valid ? exp(logit - logit_max) / totals.y : 0.0f;
  // Each child's pool index (pool tokens are distinct), visible after the pool encoder's exchange.
  for (uint child = 0; child < children; ++child) {
    if (owner && token == child_tokens[child]) {
      indices[child] = candidate;
    }
  }

  // 7. Encode [log p_draft, log p_corrected - log p_draft, Gumbel] with Linear -> SiLU -> Linear.
  // Both copies compute the complete first layer. Copy 0 computes second-layer channels
  // 0..7, copy 1 computes 8..15; only channels below pool_hidden are consumed.
  // The log-probability inputs above have already been floored; their difference is also a feature.
  float first[POOL_HIDDEN_MAX];
  METAL_PRAGMA_UNROLL
  for (uint unit = 0; unit < POOL_HIDDEN_MAX; ++unit) {
    const uint entry = unit * CH_POOL_INPUTS;
    first[unit] = valid && unit < pool_hidden
                      ? activate_silu(
                            pool_weights[entry] * log_draft + pool_weights[entry + 1] * (log_logit - log_draft) +
                            pool_weights[entry + 2] * gumbel +
                            pool_weights[layout.pool_entry_biases - layout.pool_entry_weights + unit]
                        )
                      : 0.0f;
  }
  float mean[POOL_HALF], peak[POOL_HALF];
  METAL_PRAGMA_UNROLL
  for (uint offset = 0; offset < POOL_HALF; ++offset) {
    const uint unit = half_index * POOL_HALF + offset;
    float value = -INFINITY;
    if (valid && unit < pool_hidden) {
      value = pool_weights[layout.pool_hidden_biases - layout.pool_entry_weights + unit];
      METAL_PRAGMA_UNROLL
      for (uint quad = 0; quad < POOL_HIDDEN_MAX / 4u; ++quad) {
        const float4 inputs =
            float4(first[4u * quad], first[4u * quad + 1u], first[4u * quad + 2u], first[4u * quad + 3u]);
        value += dot(pool_hidden_rows[unit * (POOL_HIDDEN_MAX / 4u) + quad], inputs);
      }
    }
    mean[offset] = valid ? weight * value : 0.0f;
    peak[offset] = value;
  }

  // 8. Reduce weighted means and componentwise maxima within SIMD groups,
  // then combine them across the threadgroup into summary = [weighted_mean; maximum].
  // The simdgroup's sums and maxima of this copy's hidden units, eight of each, by one butterfly each.
  static_assert(POOL_HALF == 8u, "the butterflies reduce eight values");
  const uint group_lane = thread_context.simd_lane_id;
  const uint offset = confidence_head_reduce8_index(group_lane);
  const float group_mean = confidence_head_reduce8<false>(mean, group_lane);
  const float group_peak = confidence_head_reduce8<true>(peak, group_lane);
  if ((group_lane & 3u) == 0u) {
    partial_mean[thread_context.simdgroup_index * POOL_HALF + offset] = group_mean;
    partial_peak[thread_context.simdgroup_index * POOL_HALF + offset] = group_peak;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid < pool_hidden) {
    // Unit `lid` belongs to one copy of the pool, that copy's simdgroups hold its partial sums.
    const uint owner_half = lid / POOL_HALF;
    const uint offset = lid % POOL_HALF;
    const uint groups = CANDIDATES_MAX / METAL_SIMD_SIZE;
    float total = 0.0f;
    float top = -INFINITY;
    for (uint group = owner_half * groups; group < (owner_half + 1u) * groups; ++group) {
      total += partial_mean[group * POOL_HALF + offset];
      top = max(top, partial_peak[group * POOL_HALF + offset]);
    }
    summary[lid] = total;
    summary[pool_hidden + lid] = top;
  }

  // 9. Count candidates whose draft logits are strictly greater than each child's draft logit.
  // The drafter ranks of the children, by the summary's exchange: each simdgroup's four counts into shared4.
  float4 above = 0.0f;
  for (uint child = 0; child < children; ++child) {
    above[child] = owner && draft > candidate_logits[base + indices[child]] ? 1.0f : 0.0f;
  }
  const uint shared_lane = thread_context.simd_lane_id;
  const float group_above = confidence_head_reduce4<false>(above, shared_lane);
  if ((shared_lane & 7u) == 0u) {
    const uint shared_index = thread_context.simdgroup_index * 4u + confidence_head_reduce4_index(shared_lane);
    reinterpret_cast<threadgroup float*>(shared4)[shared_index] = group_above;
  }

  // Defensive fallback matching the CPU kernel: an unmatched child retains index zero and uses the first pool token's
  // vector. Valid Weaver output must not need this fallback, it does not repair the unmatched token ID stored in the
  // frontier.
  for (uint position = lid; position < children * hidden; position += LOCAL_THREADS) {
    const uint child = position / hidden;
    const uint unit = position % hidden;
    const uint pool_token = candidate_ids[base + indices[child]];
    if (pool_token != child_tokens[child]) {
      vectors[position] = float(token_table[pool_token * hidden + unit]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 10. Build the node vector z = Linear(summary) + draft_rows[pool_row] and save it for prune.
  // z_u: the pool summary through its output layer, plus the drafter row's term.
  confidence_head_dense_rows<LOCAL_THREADS>(
      weights + layout.pool_out_weights,
      weights + layout.pool_out_biases,
      hidden,
      2u * pool_hidden,
      summary,
      0u,
      1u,
      node,
      0u,
      thread_context
  );
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid < hidden) {
    node[lid] += draft_rows[pool_row * hidden + lid];
    nodes[slot * hidden + lid] = node[lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 11. Predict capped softplus coefficients alpha and beta from z, controlling the contributions of corrected logits
  // and target noise.
  confidence_head_alpha_beta<LOCAL_THREADS>(node, weights, layout, hidden, cap, scalars, thread_context, lid);

  // 12. Form [log p_draft, log p_corrected, Gumbel, log(1 + rank)] for each child.
  // Standardize each value with the stored training statistics and feature mask.
  // Slot numbers: log pi_a(c), log pi_l(c), G_c from the candidate's thread, the drafter ranks from the simdgroups'
  // counts.
  const float4 ranks =
      simd_sum(thread_context.simd_lane_id < LOCAL_SIMDGROUPS ? shared4[thread_context.simd_lane_id] : float4(0.0f));
  for (uint child = 0; child < children; ++child) {
    if (owner && indices[child] == candidate) {
      const float raw[CH_SLOT] = {log_draft, log_logit, gumbel, log(1.0f + ranks[child])};
      for (uint number = 0; number < CH_SLOT; ++number) {
        numbers[child * CH_SLOT + number] = (raw[number] - weights[layout.slot_means + number]) /
                                            weights[layout.slot_stds + number] * weights[layout.slot_keep + number];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 13. Save token vectors, scalar features, pool indices and presence flags for prune.
  for (uint position = lid; position < children * (hidden + CH_SLOT); position += LOCAL_THREADS) {
    const uint child = position / (hidden + CH_SLOT);
    const uint column = position % (hidden + CH_SLOT);
    const uint inputs_index = (slot * children + child) * (hidden + CH_SLOT) + column;
    child_inputs[inputs_index] =
        column < hidden ? vectors[child * hidden + column] : numbers[child * CH_SLOT + column - hidden];
  }
  if (lid < children) {
    child_indices[slot * children + lid] = indices[lid];
    child_real[slot * children + lid] = 1u;
  }

  // 14. Predict each selected child's correction delta from [z; token vector;
  // elementwise z * token vector; scalar features] using the edge MLP.
  confidence_head_edge_deltas<LOCAL_THREADS>(
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

  // 15. Normalize alpha * logit + beta * Gumbel + delta over the WHOLE pool.
  // Only selected children receive delta; their edge scores are min(log q, 0).
  confidence_head_child_edges<LOCAL_THREADS>(
      logit,
      gumbel,
      owner,
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

  // 16. Add each edge score to the parent path score and update frontier priorities.
  // These priorities determine which branches are expanded in subsequent rounds.
  if (lid < children) {
    const float edge = edges[lid];
    const uint entry = slot * children + lid;
    const float path = as_type<float>(packed_tree[uint(TreeIdx::PathLogprobBits) * tree_slot_count + slot]) + edge;
    child_edges_out[row * children + lid] = edge;
    frontier[uint(FrontierIdx::PathLogprobBits) * frontier_capacity + entry] = as_type<uint>(path);
    frontier[uint(FrontierIdx::EdgeLogprobBits) * frontier_capacity + entry] = as_type<uint>(edge);
    frontier[uint(FrontierIdx::PathScoreKey) * frontier_capacity + entry] = top_k_score_key(path);
  }
}

#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/top_k.h"
#include "../weaver/weaver_frontier.h"
#include "confidence_head.h"

using namespace metal;

static_assert(CH_POOL_COPIES * CH_CANDIDATES_MAX == CH_LOCAL_THREADS, "every copy of the pool has a thread per candidate");

// The head's local part on the nodes one round expanded, right after Weaver's top_children put their children into the
// frontier (entry slot * children + j for child j of the node in tree slot `slot`): for every child the edge
// min(log q_u(c), 0), which replaces the model's in the frontier entry, so that its path is the parent's plus the edge
// (log r) and its priority follows; and the node's rows of the step state the final prune reads. One threadgroup per
// node; the CPU twin is the reference.
PUBLIC KERNEL(ConfidenceHeadRound)(
    const device bfloat* residual_logits,
    const device float* candidate_logits,
    const device uint* candidate_ids,
    const device float* noise,
    const device float* draft_rows,
    const device uint* node_metadata,
    const device uint* node_valid,
    const device uint* packed_tree,
    const device bfloat* token_table,
    const device float* local_weights,
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
    threadgroup float shared[2 * CH_LOCAL_SIMDGROUPS],
    threadgroup float4 shared4[CH_LOCAL_SIMDGROUPS],
    threadgroup float pool_weights[CH_POOL_WEIGHTS_MAX],
    threadgroup float4 pool_hidden_rows[CH_POOL_HIDDEN_MAX * CH_POOL_HIDDEN_MAX / 4],
    threadgroup float partial_mean[CH_LOCAL_SIMDGROUPS * CH_POOL_HALF],
    threadgroup float partial_peak[CH_LOCAL_SIMDGROUPS * CH_POOL_HALF],
    threadgroup float summary[2 * CH_POOL_HIDDEN_MAX],
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
    const uint lid THREADS(CH_LOCAL_THREADS)
) {
  if (node_valid[row] == 0u) {
    return;
  }
  const ConfidenceHeadLayout layout = confidence_head_layout(hidden, pool_hidden, edge_hidden);
  const device float* weights = local_weights;
  const uint depth = node_metadata[uint(MetadataIdx::Depth) * rows + row];
  const uint slot = node_metadata[uint(MetadataIdx::TreeSlot) * rows + row];
  const uint pool_row = min(depth, pool_rows - 1u);
  const uint base = row * candidates;
  const device uint* child_token_ids = frontier + uint(FrontierIdx::TokenId) * frontier_capacity + slot * children;

  // The pool encoder's weights (entry layer, its biases, hidden layer, its biases: contiguous in the packed buffer),
  // read by every thread, go to threadgroup memory once.
  const uint pool_weight_count = pool_hidden * (CH_POOL_INPUTS + 1u + pool_hidden + 1u);
  for (uint index = lid; index < pool_weight_count; index += CH_LOCAL_THREADS) {
    pool_weights[index] = weights[layout.pool_entry_weights + index];
  }
  // The hidden layer's rows again, padded with zeros to CH_POOL_HIDDEN_MAX columns: read as float4.
  for (uint index = lid; index < CH_POOL_HIDDEN_MAX * CH_POOL_HIDDEN_MAX; index += CH_LOCAL_THREADS) {
    const uint unit = index / CH_POOL_HIDDEN_MAX, inner = index % CH_POOL_HIDDEN_MAX;
    reinterpret_cast<threadgroup float*>(pool_hidden_rows)[index] =
        unit < pool_hidden && inner < pool_hidden ? weights[layout.pool_hidden_weights + unit * pool_hidden + inner]
                                                  : 0.0f;
  }

  // The children's tokens and token vectors need nothing computed: they go first, and the softmax's exchange makes them
  // visible.
  if (lid < children) {
    child_tokens[lid] = child_token_ids[lid];
    indices[lid] = 0u;
    real[lid] = 1u;
  }
  for (uint position = lid; position < children * hidden; position += CH_LOCAL_THREADS) {
    const uint child = position / hidden, unit = position % hidden;
    vectors[position] = float(token_table[child_token_ids[child] * hidden + unit]);
  }

  // This thread's candidate; every copy of the pool holds it, the first copy counts it in the pool's sums.
  const uint candidate = lid % CH_CANDIDATES_MAX, half_index = lid / CH_CANDIDATES_MAX;
  const bool valid = candidate < candidates;
  const bool owner = valid && half_index == 0u;
  const float draft = valid ? candidate_logits[base + candidate] : -INFINITY;
  const float logit = valid ? draft + float(residual_logits[base + candidate]) : -INFINITY;
  const float gumbel = valid ? noise[pool_row * candidates + candidate] : 0.0f;
  const uint token = valid ? candidate_ids[base + candidate] : CH_NONE;
  if (owner) {
    slot_logits[slot * candidates + candidate] = logit;
  }

  // log pi_a, log pi_l (floored) and pi_l over the pool: both log-sum-exps in one exchange.
  const float4 normalisers =
      confidence_head_log_sum_exp2<CH_LOCAL_SIMDGROUPS>(float2(draft, logit), owner, shared4, thread_context);
  const float draft_max = normalisers.x, logit_max = normalisers.y;
  const float2 totals = normalisers.zw;
  const float log_draft = max((draft - draft_max) - log(totals.x), CH_LOG_PROB_FLOOR);
  const float log_logit = max((logit - logit_max) - log(totals.y), CH_LOG_PROB_FLOOR);
  const float weight = valid ? exp(logit - logit_max) / totals.y : 0.0f;
  // Each child's pool index (pool tokens are distinct), visible after the pool encoder's exchange.
  for (uint child = 0; child < children; ++child) {
    if (owner && token == child_tokens[child]) {
      indices[child] = candidate;
    }
  }

  // The pool set encoder: pi_l-weighted mean and maximum of every hidden unit over the pool. Every thread computes its
  // candidate's whole entry layer and the half of the hidden units its half of the threadgroup owns.
  float first[CH_POOL_HIDDEN_MAX];
  METAL_PRAGMA_UNROLL
  for (uint unit = 0; unit < CH_POOL_HIDDEN_MAX; ++unit) {
    const uint entry = unit * CH_POOL_INPUTS;
    first[unit] = valid && unit < pool_hidden
                      ? confidence_head_silu(
                            pool_weights[entry] * log_draft + pool_weights[entry + 1] * (log_logit - log_draft) +
                            pool_weights[entry + 2] * gumbel +
                            pool_weights[layout.pool_entry_biases - layout.pool_entry_weights + unit]
                        )
                      : 0.0f;
  }
  float mean[CH_POOL_HALF], peak[CH_POOL_HALF];
  METAL_PRAGMA_UNROLL
  for (uint offset = 0; offset < CH_POOL_HALF; ++offset) {
    const uint unit = half_index * CH_POOL_HALF + offset;
    float value = -INFINITY;
    if (valid && unit < pool_hidden) {
      value = pool_weights[layout.pool_hidden_biases - layout.pool_entry_weights + unit];
      METAL_PRAGMA_UNROLL
      for (uint quad = 0; quad < CH_POOL_HIDDEN_MAX / 4u; ++quad) {
        const float4 inputs = float4(first[4u * quad], first[4u * quad + 1u], first[4u * quad + 2u], first[4u * quad + 3u]);
        value += dot(pool_hidden_rows[unit * (CH_POOL_HIDDEN_MAX / 4u) + quad], inputs);
      }
    }
    mean[offset] = valid ? weight * value : 0.0f;
    peak[offset] = value;
  }
  // The simdgroup's sums and maxima of this copy's hidden units, eight of each, by one butterfly each.
  static_assert(CH_POOL_HALF == 8u, "the butterflies reduce eight values");
  {
    const uint lane = thread_context.simd_lane_id, offset = confidence_head_reduce8_index(lane);
    const float group_mean = confidence_head_reduce8<false>(mean, lane);
    const float group_peak = confidence_head_reduce8<true>(peak, lane);
    if ((lane & 3u) == 0u) {
      partial_mean[thread_context.simdgroup_index * CH_POOL_HALF + offset] = group_mean;
      partial_peak[thread_context.simdgroup_index * CH_POOL_HALF + offset] = group_peak;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < pool_hidden) {
    // Unit `lid` belongs to one copy of the pool; that copy's simdgroups hold its partial sums.
    const uint owner_half = lid / CH_POOL_HALF, offset = lid % CH_POOL_HALF;
    const uint groups = CH_CANDIDATES_MAX / METAL_SIMD_SIZE;
    float total = 0.0f, top = -INFINITY;
    for (uint group = owner_half * groups; group < (owner_half + 1u) * groups; ++group) {
      total += partial_mean[group * CH_POOL_HALF + offset];
      top = max(top, partial_peak[group * CH_POOL_HALF + offset]);
    }
    summary[lid] = total;
    summary[pool_hidden + lid] = top;
  }
  // The drafter ranks of the children, by the summary's exchange: each simdgroup's four counts into shared4.
  {
    float4 above = 0.0f;
    for (uint child = 0; child < children; ++child) {
      above[child] = owner && draft > candidate_logits[base + indices[child]] ? 1.0f : 0.0f;
    }
    const uint lane = thread_context.simd_lane_id;
    const float group_above = confidence_head_reduce4<false>(above, lane);
    if ((lane & 7u) == 0u) {
      reinterpret_cast<threadgroup float*>(shared4)[thread_context.simdgroup_index * 4u + confidence_head_reduce4_index(lane)] =
          group_above;
    }
  }
  // A child missing from the pool reads the vector of the pool's first token, as the CPU twin does.
  for (uint position = lid; position < children * hidden; position += CH_LOCAL_THREADS) {
    const uint child = position / hidden, unit = position % hidden;
    const uint pool_token = candidate_ids[base + indices[child]];
    if (pool_token != child_tokens[child]) {
      vectors[position] = float(token_table[pool_token * hidden + unit]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // z_u: the pool summary through its output layer, plus the drafter row's term.
  confidence_head_dense_rows<CH_LOCAL_THREADS>(
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
  confidence_head_alpha_beta<CH_LOCAL_THREADS>(node, weights, layout, hidden, cap, scalars, thread_context, lid);

  // Slot numbers: log pi_a(c), log pi_l(c), G_c from the candidate's thread, the drafter ranks from the simdgroups'
  // counts.
  const float4 ranks = simd_sum(
      thread_context.simd_lane_id < CH_LOCAL_SIMDGROUPS ? shared4[thread_context.simd_lane_id] : float4(0.0f)
  );
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
  for (uint position = lid; position < children * (hidden + CH_SLOT); position += CH_LOCAL_THREADS) {
    const uint child = position / (hidden + CH_SLOT), column = position % (hidden + CH_SLOT);
    child_inputs[(slot * children + child) * (hidden + CH_SLOT) + column] =
        column < hidden ? vectors[child * hidden + column] : numbers[child * CH_SLOT + column - hidden];
  }
  if (lid < children) {
    child_indices[slot * children + lid] = indices[lid];
    child_real[slot * children + lid] = 1u;
  }

  confidence_head_edge_deltas<CH_LOCAL_THREADS>(
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
  confidence_head_child_edges<CH_LOCAL_THREADS>(
      logit, gumbel, owner, candidate, indices, deltas, real, edges, shared, scalars, children, thread_context
  );
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

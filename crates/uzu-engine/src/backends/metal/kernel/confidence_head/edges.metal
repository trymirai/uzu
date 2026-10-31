#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/top_k.h"
#include "../weaver/weaver_frontier.h"
#include "head.h"

using namespace metal;

struct EdgesMemory {
  float reduce[2u * HEAD_SIMDGROUPS];
  float coefficients[2];
  float node[MODEL_MAX];
  float tokens[CHILDREN_MAX * MODEL_MAX];
  float numbers[CHILDREN_MAX * SLOT_NUMBERS];
  float input[CHILDREN_MAX * HEAD_EDGE_INPUT_MAX];
  float state[CHILDREN_MAX * MODEL_MAX];
  float block[CHILDREN_MAX * MODEL_MAX];
  float raised[CHILDREN_MAX * EDGE_HIDDEN_MAX];
  float deltas[CHILDREN_MAX];
  float edges[CHILDREN_MAX];
  uint pools[CHILDREN_MAX];
};

PUBLIC KERNEL(ConfidenceHeadEdges)(
    const device uint* node_metadata OPTIONAL(!refine),
    const device uint* slot_rows,
    const device bfloat* nodes,
    const device bfloat* node_shifts OPTIONAL(refine),
    const device bfloat* child_shifts OPTIONAL(refine),
    const device bfloat* token_features,
    const device float* child_numbers,
    const device uint* child_pools,
    const device float* scores,
    device uint* frontier,
    device uint* packed_tree,
    device bfloat* child_inputs OPTIONAL(!refine),
    const device bfloat* input_norm_scales,
    const device bfloat* input_norm_biases,
    const device bfloat* entry_weights,
    const device bfloat* entry_biases,
    const device bfloat* block_norm_scales,
    const device bfloat* block_norm_biases,
    const device bfloat* up_weights,
    const device bfloat* up_biases,
    const device bfloat* down_weights,
    const device bfloat* down_biases,
    const device bfloat* final_norm_scales,
    const device bfloat* final_norm_biases,
    const device bfloat* output_weights,
    const device bfloat* output_biases,
    const device bfloat* alpha_beta_weights,
    const device bfloat* alpha_beta_biases,
    const constant ConfidenceHeadParams& params,
    const constant uint& groups,
    const uint model_dim SPECIALIZE,
    const uint edge_hidden SPECIALIZE,
    const bool refine SPECIALIZE,
    threadgroup EdgesMemory& memory,
    const ThreadContext context,
    const uint group GROUPS(groups),
    const uint lid THREADS(HEAD_THREADS)
) {
  const uint tree_slot_count = params.tree_slot_count;
  const uint slot = refine ? group : node_metadata[uint(MetadataIdx::TreeSlot) * groups + group];
  if (slot >= tree_slot_count || slot_rows[slot] == HEAD_NONE) {
    return;
  }
  const uint row = slot_rows[slot];
  const uint children = params.children;
  const uint candidates = params.candidates;
  const uint capacity = tree_slot_count * children;
  const uint first_entry = slot * children;
  if (lid < model_dim) {
    memory.node[lid] =
        float(nodes[slot * model_dim + lid]) + (refine ? float(node_shifts[row * model_dim + lid]) : 0.0f);
  }
  if (lid < children) {
    memory.pools[lid] = child_pools[first_entry + lid];
  }
  for (uint index = lid; index < children * model_dim; index += HEAD_THREADS) {
    const uint child = index / model_dim;
    const uint unit = index % model_dim;
    const uint entry = first_entry + child;
    const uint token = frontier[uint(FrontierIdx::TokenId) * capacity + entry];
    memory.tokens[child * MODEL_MAX + unit] =
        float(token_features[token * model_dim + unit]) +
        (refine ? float(child_shifts[(1 + entry) * model_dim + unit]) : 0.0f);
  }
  if (lid < children * SLOT_NUMBERS) {
    memory.numbers[lid] = child_numbers[first_entry * SLOT_NUMBERS + lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint inputs = 3u * model_dim + SLOT_NUMBERS;
  for (uint index = lid; index < children * inputs; index += HEAD_THREADS) {
    const uint child = index / inputs;
    const uint column = index % inputs;
    const uint unit = column % model_dim;
    const float token = memory.tokens[child * MODEL_MAX + unit];
    float value;
    if (column < model_dim) {
      value = memory.node[unit];
    } else if (column < 2u * model_dim) {
      value = token;
    } else if (column < 3u * model_dim) {
      value = memory.node[unit] * token;
    } else {
      value = memory.numbers[child * SLOT_NUMBERS + column - 3u * model_dim];
    }
    memory.input[child * HEAD_EDGE_INPUT_MAX + column] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  head_layer_norm(
      memory.input, memory.input, HEAD_EDGE_INPUT_MAX, inputs, children, input_norm_scales, input_norm_biases,
      params.epsilon, context
  );
  head_dense(
      entry_weights, entry_biases, memory.input, HEAD_EDGE_INPUT_MAX, memory.state, MODEL_MAX, inputs, model_dim,
      children, false, false, context
  );
  head_layer_norm(
      memory.state, memory.block, MODEL_MAX, model_dim, children, block_norm_scales, block_norm_biases,
      params.epsilon, context
  );
  head_dense(
      up_weights, up_biases, memory.block, MODEL_MAX, memory.raised, EDGE_HIDDEN_MAX, model_dim, edge_hidden,
      children, true, false, context
  );
  head_dense(
      down_weights, down_biases, memory.raised, EDGE_HIDDEN_MAX, memory.state, MODEL_MAX, edge_hidden,
      model_dim, children, false, true, context
  );
  head_layer_norm(
      memory.state, memory.state, MODEL_MAX, model_dim, children, final_norm_scales, final_norm_biases,
      params.epsilon, context
  );
  const uint simdgroup = context.simdgroup_index;
  const uint lane = context.simd_lane_id;
  if (simdgroup < children) {
    const float delta = head_simd_dot(output_weights, memory.state + simdgroup * MODEL_MAX, model_dim, lane);
    if (lane == 0) {
      memory.deltas[simdgroup] = delta + float(output_biases[0]);
    }
  } else if (simdgroup >= CHILDREN_MAX && simdgroup < CHILDREN_MAX + 2u) {
    const uint output = simdgroup - CHILDREN_MAX;
    const float value =
        head_simd_dot(alpha_beta_weights + output * model_dim, memory.node, model_dim, lane) + float(alpha_beta_biases[output]);
    if (lane == 0) {
      memory.coefficients[output] = min(activate_softplus(value), params.logit_cap);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool valid = lid < candidates;
  float score = valid ? memory.coefficients[0] * scores[(2 * slot) * candidates + lid] +
                            memory.coefficients[1] * scores[(2 * slot + 1) * candidates + lid]
                      : -INFINITY;
  for (uint child = 0; child < children; ++child) {
    score += memory.pools[child] == lid ? memory.deltas[child] : 0.0f;
  }
  const float log_sum = head_log_sum_exp(score, memory.reduce, context);
  const float parent_path = as_type<float>(packed_tree[uint(TreeIdx::PathLogprobBits) * tree_slot_count + slot]);
  for (uint child = 0; child < children; ++child) {
    if (memory.pools[child] == lid) {
      const uint entry = first_entry + child;
      const float edge = min(score - log_sum, 0.0f);
      memory.edges[child] = edge;
      frontier[uint(FrontierIdx::EdgeLogprobBits) * capacity + entry] = as_type<uint>(edge);
      if (!refine) {
        frontier[uint(FrontierIdx::PathLogprobBits) * capacity + entry] = as_type<uint>(parent_path + edge);
        frontier[uint(FrontierIdx::PathScoreKey) * capacity + entry] = top_k_score_key(parent_path + edge);
      }
    }
  }
  if (!refine) {
    const uint width = model_dim + SLOT_NUMBERS;
    for (uint index = lid; index < children * width; index += HEAD_THREADS) {
      const uint child = index / width;
      const uint column = index % width;
      child_inputs[(first_entry + child) * width + column] =
          bfloat(column < model_dim ? memory.tokens[child * MODEL_MAX + column]
                                    : memory.numbers[child * SLOT_NUMBERS + column - model_dim]);
    }
    return;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint other = lid; other < tree_slot_count; other += HEAD_THREADS) {
    if (packed_tree[uint(TreeIdx::ParentSlot) * tree_slot_count + other] == slot) {
      const uint token = packed_tree[uint(TreeIdx::TokenId) * tree_slot_count + other];
      for (uint child = 0; child < children; ++child) {
        if (frontier[uint(FrontierIdx::TokenId) * capacity + first_entry + child] == token) {
          packed_tree[uint(TreeIdx::EdgeLogprobBits) * tree_slot_count + other] = as_type<uint>(memory.edges[child]);
        }
      }
    }
  }
}

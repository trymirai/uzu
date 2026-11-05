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
    const device uint* node_valid OPTIONAL(!refine),
    const device uint* candidate_ids OPTIONAL(!refine),
    const device float* pool_features OPTIONAL(!refine),
    const device bfloat* slot_stats OPTIONAL(!refine),
    device uint* slot_rows,
    device uint* row_slots OPTIONAL(!refine),
    const device bfloat* nodes,
    const device bfloat* node_shifts OPTIONAL(refine),
    const device bfloat* child_shifts OPTIONAL(refine),
    const device bfloat* token_features,
    device float* child_numbers,
    device uint* child_pools,
    const device float* scores,
    device uint* frontier,
    device uint* packed_tree,
    device bfloat* child_inputs OPTIONAL(!refine),
    const device bfloat* input_norm,
    const device bfloat* entry_weights,
    const device bfloat* entry_biases,
    const device bfloat* block_norm,
    const device bfloat* up_weights,
    const device bfloat* up_biases,
    const device bfloat* down_weights,
    const device bfloat* down_biases,
    const device bfloat* final_norm,
    const device bfloat* readout_weights,
    const device bfloat* readout_biases,
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
  const uint children = params.children;
  const uint candidates = params.candidates;
  const uint capacity = tree_slot_count * children;
  const uint slot = refine ? group : node_metadata[uint(MetadataIdx::TreeSlot) * groups + group];
  if (slot >= tree_slot_count || (refine ? slot_rows[slot] == HEAD_NONE : node_valid[group] == 0u)) {
    return;
  }
  const uint first_entry = slot * children;
  const uint simdgroup = context.simdgroup_index;
  const uint lane = context.simd_lane_id;
  const uint inputs = 3u * model_dim + SLOT_NUMBERS;
  const uint row = refine ? slot_rows[slot] : 0u;
  if (refine) {
    if (lid < children) {
      memory.pools[lid] = child_pools[first_entry + lid];
    }
    if (lid < children * SLOT_NUMBERS) {
      memory.numbers[lid] = child_numbers[first_entry * SLOT_NUMBERS + lid];
    }
  } else {
    uint child_tokens[CHILDREN_MAX];
    METAL_PRAGMA_UNROLL
    for (uint child = 0; child < CHILDREN_MAX; ++child) {
      child_tokens[child] =
          child < children ? frontier[uint(FrontierIdx::TokenId) * capacity + first_entry + child] : HEAD_NONE;
    }
    if (lid < candidates) {
      const uint depth = node_metadata[uint(MetadataIdx::Depth) * groups + group];
      const uint candidate_id = candidate_ids[depth * candidates + lid];
      for (uint child = 0; child < children; ++child) {
        if (candidate_id == child_tokens[child]) {
          const uint entry = first_entry + child;
          const device float* features = pool_features + depth * POOL_FEATURES * candidates;
          const float numbers[SLOT_NUMBERS] = {
              features[lid],
              max(scores[(2 * slot) * candidates + lid], HEAD_LOG_PROB_FLOOR),
              scores[(2 * slot + 1) * candidates + lid],
              features[2 * candidates + lid]
          };
          memory.pools[child] = lid;
          child_pools[entry] = lid;
          for (uint number = 0; number < SLOT_NUMBERS; ++number) {
            const float value =
                (numbers[number] - float(slot_stats[number])) / float(slot_stats[SLOT_NUMBERS + number]);
            memory.numbers[child * SLOT_NUMBERS + number] = value;
            child_numbers[entry * SLOT_NUMBERS + number] = value;
          }
        }
      }
    }
    if (simdgroup == 0) {
      uint own_row = 0;
      if (slot != 0) {
        const uint parent = packed_tree[uint(TreeIdx::ParentSlot) * tree_slot_count + slot];
        const uint own_token = packed_tree[uint(TreeIdx::TokenId) * tree_slot_count + slot];
        const bool hit =
            lane < children && frontier[uint(FrontierIdx::TokenId) * capacity + parent * children + lane] == own_token;
        own_row = simd_max(hit ? 1 + parent * children + lane : 0u);
      }
      if (lane == 0) {
        slot_rows[slot] = own_row;
        row_slots[own_row] = slot;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup < children) {
    const uint entry = first_entry + simdgroup;
    const uint token = frontier[uint(FrontierIdx::TokenId) * capacity + entry];
    threadgroup float* line = memory.input + simdgroup * HEAD_EDGE_INPUT_MAX;
    float total = 0.0f;
    for (uint unit = lane; unit < model_dim; unit += METAL_SIMD_SIZE) {
      const float node =
          float(nodes[slot * model_dim + unit]) + (refine ? float(node_shifts[row * model_dim + unit]) : 0.0f);
      const float vector = float(token_features[token * model_dim + unit]) +
                           (refine ? float(child_shifts[(1 + entry) * model_dim + unit]) : 0.0f);
      line[unit] = node;
      line[model_dim + unit] = vector;
      line[2u * model_dim + unit] = node * vector;
      memory.tokens[simdgroup * MODEL_MAX + unit] = vector;
      if (simdgroup == 0) {
        memory.node[unit] = node;
      }
      total += node + vector + node * vector;
    }
    if (lane < SLOT_NUMBERS) {
      const float number = memory.numbers[simdgroup * SLOT_NUMBERS + lane];
      line[3u * model_dim + lane] = number;
      total += number;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = simd_sum(total) / float(inputs);
    float spread = 0.0f;
    for (uint column = lane; column < inputs; column += METAL_SIMD_SIZE) {
      spread += (line[column] - mean) * (line[column] - mean);
    }
    const float scale = rsqrt(simd_sum(spread) / float(inputs) + params.epsilon);
    for (uint column = lane; column < inputs; column += METAL_SIMD_SIZE) {
      line[column] = (line[column] - mean) * scale * float(input_norm[column]) + float(input_norm[inputs + column]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  head_dense(
      entry_weights,
      entry_biases,
      memory.input,
      HEAD_EDGE_INPUT_MAX,
      memory.state,
      MODEL_MAX,
      inputs,
      model_dim,
      children,
      false,
      false,
      context
  );
  head_layer_norm(
      memory.state,
      memory.block,
      MODEL_MAX,
      model_dim,
      children,
      block_norm,
      block_norm + model_dim,
      params.epsilon,
      context
  );
  head_dense(
      up_weights,
      up_biases,
      memory.block,
      MODEL_MAX,
      memory.raised,
      EDGE_HIDDEN_MAX,
      model_dim,
      edge_hidden,
      children,
      true,
      false,
      context
  );
  head_dense(
      down_weights,
      down_biases,
      memory.raised,
      EDGE_HIDDEN_MAX,
      memory.state,
      MODEL_MAX,
      edge_hidden,
      model_dim,
      children,
      false,
      true,
      context
  );
  if (simdgroup < children) {
    const threadgroup float* line = memory.state + simdgroup * MODEL_MAX;
    float total = 0.0f;
    for (uint unit = lane; unit < model_dim; unit += METAL_SIMD_SIZE) {
      total += line[unit];
    }
    const float mean = simd_sum(total) / float(model_dim);
    float spread = 0.0f;
    for (uint unit = lane; unit < model_dim; unit += METAL_SIMD_SIZE) {
      spread += (line[unit] - mean) * (line[unit] - mean);
    }
    const float scale = rsqrt(simd_sum(spread) / float(model_dim) + params.epsilon);
    float delta = 0.0f;
    for (uint unit = lane; unit < model_dim; unit += METAL_SIMD_SIZE) {
      const float normed = (line[unit] - mean) * scale * float(final_norm[unit]) + float(final_norm[model_dim + unit]);
      delta += float(readout_weights[unit]) * normed;
    }
    delta = simd_sum(delta);
    if (lane == 0) {
      memory.deltas[simdgroup] = delta + float(readout_biases[0]);
    }
  } else if (simdgroup >= CHILDREN_MAX && simdgroup < CHILDREN_MAX + 2u) {
    const uint output = simdgroup - CHILDREN_MAX;
    const float value = head_simd_dot(readout_weights + (1 + output) * model_dim, memory.node, model_dim, lane) +
                        float(readout_biases[1 + output]);
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
      child_inputs[(first_entry + child) * width + column] = bfloat(
          column < model_dim ? memory.tokens[child * MODEL_MAX + column]
                             : memory.numbers[child * SLOT_NUMBERS + column - model_dim]
      );
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

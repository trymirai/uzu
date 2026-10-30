#include <metal_stdlib>
#include "../common/dsl.h"
#include "../weaver/weaver_frontier.h"
#include "head.h"

using namespace metal;

#define LOG_PROB_FLOOR (-30.0f)

struct NodesMemory {
  float4 pool_entry[HEAD_POOL_HIDDEN_MAX];
  float4 pool_hidden[HEAD_POOL_HIDDEN_MAX * HEAD_POOL_HIDDEN_MAX / 4u];
  float pool_hidden_biases[HEAD_POOL_HIDDEN_MAX];
  float reduce[2u * HEAD_SIMDGROUPS * HEAD_POOL_HIDDEN_MAX];
  float summary[2u * HEAD_POOL_HIDDEN_MAX];
  float node[HEAD_MODEL_MAX];
};

PUBLIC KERNEL(ConfidenceHeadNodes)(
    const device bfloat* residual_logits,
    const device float* candidate_logits,
    const device uint* candidate_ids,
    const device float* pool_features,
    const device uint* node_metadata,
    const device uint* node_valid,
    const device uint* packed_tree,
    const device uint* frontier,
    const device bfloat* draft,
    const device bfloat* pool_entry_weights,
    const device bfloat* pool_entry_biases,
    const device bfloat* pool_hidden_weights,
    const device bfloat* pool_hidden_biases,
    const device bfloat* pool_out_weights,
    const device bfloat* pool_out_biases,
    const device bfloat* slot_mean,
    const device bfloat* slot_std,
    device float* scores,
    device bfloat* nodes,
    device bfloat* node_inputs,
    device uint* child_pools,
    device float* child_numbers,
    device uint* slot_rows,
    device uint* row_slots,
    const constant ConfidenceHeadParams& params,
    const constant uint& rows,
    const uint model_dim SPECIALIZE,
    const uint pool_hidden SPECIALIZE,
    threadgroup NodesMemory& memory,
    const ThreadContext context,
    const uint row GROUPS(rows),
    const uint lid THREADS(HEAD_THREADS)
) {
  const uint slot = node_metadata[uint(MetadataIdx::TreeSlot) * rows + row];
  if (node_valid[row] == 0u || slot >= params.tree_slot_count) {
    return;
  }
  const uint depth = node_metadata[uint(MetadataIdx::Depth) * rows + row];
  const uint candidates = params.candidates;
  const uint capacity = params.tree_slot_count * params.children;
  const uint first_entry = slot * params.children;
  const device float* features = pool_features + depth * 3 * candidates;
  const bool valid = lid < candidates;
  const float logit =
      valid ? candidate_logits[depth * candidates + lid] + float(residual_logits[row * candidates + lid]) : -INFINITY;
  const float log_draft = valid ? features[lid] : 0.0f;
  const float noise = valid ? features[candidates + lid] : 0.0f;
  if (valid) {
    scores[(2 * slot) * candidates + lid] = logit;
    scores[(2 * slot + 1) * candidates + lid] = noise;
  }
  if (lid < HEAD_POOL_HIDDEN_MAX) {
    const bool used = lid < pool_hidden;
    memory.pool_entry[lid] = used ? float4(
                                        float(pool_entry_weights[3u * lid]),
                                        float(pool_entry_weights[3u * lid + 1u]),
                                        float(pool_entry_weights[3u * lid + 2u]),
                                        float(pool_entry_biases[lid])
                                    )
                                  : float4(0.0f);
    memory.pool_hidden_biases[lid] = used ? float(pool_hidden_biases[lid]) : 0.0f;
  }
  for (uint index = lid; index < HEAD_POOL_HIDDEN_MAX * HEAD_POOL_HIDDEN_MAX; index += HEAD_THREADS) {
    const uint unit = index / HEAD_POOL_HIDDEN_MAX;
    const uint inner = index % HEAD_POOL_HIDDEN_MAX;
    reinterpret_cast<threadgroup float*>(memory.pool_hidden)[index] =
        unit < pool_hidden && inner < pool_hidden ? float(pool_hidden_weights[unit * pool_hidden + inner]) : 0.0f;
  }
  if (lid < model_dim) {
    memory.node[lid] = float(draft[depth * model_dim + lid]);
  }

  const float log_logit = logit - head_log_sum_exp(float2(logit, -INFINITY), memory.reduce, context).x;
  const float floored = max(log_logit, LOG_PROB_FLOOR);
  const float weight = valid ? exp(log_logit) : 0.0f;
  const float4 inputs = float4(log_draft, floored - log_draft, noise, 1.0f);
  float4 hidden[HEAD_POOL_HIDDEN_MAX / 4u];
  METAL_PRAGMA_UNROLL
  for (uint unit = 0; unit < HEAD_POOL_HIDDEN_MAX; ++unit) {
    const float value = dot(memory.pool_entry[unit], inputs);
    hidden[unit / 4u][unit % 4u] = unit < pool_hidden ? value / (1.0f + exp(-value)) : 0.0f;
  }
  METAL_PRAGMA_UNROLL
  for (uint unit = 0; unit < HEAD_POOL_HIDDEN_MAX; ++unit) {
    if (unit >= pool_hidden) {
      break;
    }
    float value = memory.pool_hidden_biases[unit];
    METAL_PRAGMA_UNROLL
    for (uint quad = 0; quad < HEAD_POOL_HIDDEN_MAX / 4u; ++quad) {
      value += dot(memory.pool_hidden[unit * (HEAD_POOL_HIDDEN_MAX / 4u) + quad], hidden[quad]);
    }
    const float mean = simd_sum(weight * value);
    const float peak = simd_max(valid ? value : -INFINITY);
    if (context.simd_lane_id == 0) {
      memory.reduce[(2 * context.simdgroup_index) * pool_hidden + unit] = mean;
      memory.reduce[(2 * context.simdgroup_index + 1) * pool_hidden + unit] = peak;
    }
  }
  for (uint child = 0; child < params.children; ++child) {
    const uint entry = first_entry + child;
    if (valid && candidate_ids[depth * candidates + lid] == frontier[uint(FrontierIdx::TokenId) * capacity + entry]) {
      const float numbers[HEAD_SLOT_NUMBERS] = {log_draft, floored, noise, features[2 * candidates + lid]};
      child_pools[entry] = lid;
      for (uint number = 0; number < HEAD_SLOT_NUMBERS; ++number) {
        child_numbers[entry * HEAD_SLOT_NUMBERS + number] = (numbers[number] - float(slot_mean[number])) / float(slot_std[number]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 2u * pool_hidden) {
    const bool is_mean = lid < pool_hidden;
    float value = is_mean ? 0.0f : -INFINITY;
    for (uint group = 0; group < HEAD_SIMDGROUPS; ++group) {
      const float part = memory.reduce[(2 * group + (is_mean ? 0u : 1u)) * pool_hidden + lid % pool_hidden];
      value = is_mean ? value + part : max(value, part);
    }
    memory.summary[lid] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  head_dense(
      pool_out_weights, pool_out_biases, memory.summary, 0, memory.node, 0, 2u * pool_hidden, model_dim, 1, false, true, context
  );
  if (lid < model_dim) {
    nodes[slot * model_dim + lid] = bfloat(memory.node[lid]);
    node_inputs[slot * model_dim + lid] = bfloat(memory.node[lid]);
  }
  if (lid == 0) {
    uint own_row = 0;
    if (slot != 0) {
      const uint parent = packed_tree[uint(TreeIdx::ParentSlot) * params.tree_slot_count + slot];
      const uint own_token = packed_tree[uint(TreeIdx::TokenId) * params.tree_slot_count + slot];
      for (uint child = 0; child < params.children; ++child) {
        if (frontier[uint(FrontierIdx::TokenId) * capacity + parent * params.children + child] == own_token) {
          own_row = 1 + parent * params.children + child;
        }
      }
    }
    slot_rows[slot] = own_row;
    row_slots[own_row] = slot;
  }
}

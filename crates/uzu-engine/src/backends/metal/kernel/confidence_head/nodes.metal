#include <metal_stdlib>
#include "../common/dsl.h"
#include "../weaver/weaver_frontier.h"
#include "head.h"

using namespace metal;

#define NODES_THREADS 1024u
#define NODES_SIMDGROUPS (NODES_THREADS / 32u)
#define NODES_UNITS (POOL_HIDDEN_MAX / 2u)

static_assert(NODES_THREADS == 2u * CANDIDATES_MAX, "two threads per candidate, each with half of the pool units");
static_assert(2u * POOL_HIDDEN_MAX <= NODES_SIMDGROUPS, "a simdgroup per pool summary value");

struct NodesMemory {
  float4 pool_entry[POOL_HIDDEN_MAX];
  float4 pool_hidden[POOL_HIDDEN_MAX * POOL_HIDDEN_MAX / 4u];
  float pool_hidden_biases[POOL_HIDDEN_MAX];
  float reduce[2u * NODES_SIMDGROUPS * NODES_UNITS];
  float summary[2u * POOL_HIDDEN_MAX];
  float node[MODEL_MAX];
};

PUBLIC KERNEL(ConfidenceHeadNodes)(
    const device bfloat* residual_logits,
    const device float* candidate_logits,
    const device float* pool_features,
    const device uint* round_metadata,
    const device uint* round_valid,
    const device bfloat* draft,
    const device bfloat* pool_entry_weights,
    const device bfloat* pool_entry_biases,
    const device bfloat* pool_hidden_weights,
    const device bfloat* pool_hidden_biases,
    const device bfloat* pool_out_weights,
    const device bfloat* pool_out_biases,
    device float* scores,
    device bfloat* nodes,
    device bfloat* node_inputs,
    const constant ConfidenceHeadParams& params,
    const constant uint& rows,
    const uint model_dim SPECIALIZE,
    const uint pool_hidden SPECIALIZE,
    threadgroup NodesMemory& memory,
    const ThreadContext context,
    const uint row GROUPS(rows),
    const uint lid THREADS(NODES_THREADS)
) {
  const uint slot = round_metadata[uint(MetadataIdx::TreeSlot) * rows + row];
  if (round_valid[row] == 0u || slot >= params.tree_slot_count) {
    return;
  }
  const uint depth = round_metadata[uint(MetadataIdx::Depth) * rows + row];
  const uint candidates = params.candidates;
  const device float* features = pool_features + depth * POOL_FEATURES * candidates;
  const uint candidate = lid % CANDIDATES_MAX;
  const uint half_index = lid / CANDIDATES_MAX;
  const bool valid = candidate < candidates;
  const float logit =
      valid ? candidate_logits[depth * candidates + candidate] + float(residual_logits[row * candidates + candidate])
            : -INFINITY;
  const float log_draft = valid ? features[candidate] : 0.0f;
  const float noise = valid ? features[candidates + candidate] : 0.0f;
  if (lid < POOL_HIDDEN_MAX) {
    const bool used = lid < pool_hidden;
    memory.pool_entry[lid] = used ? float4(
                                        float(pool_entry_weights[POOL_INPUTS * lid]),
                                        float(pool_entry_weights[POOL_INPUTS * lid + 1u]),
                                        float(pool_entry_weights[POOL_INPUTS * lid + 2u]),
                                        float(pool_entry_biases[lid])
                                    )
                                  : float4(0.0f);
    memory.pool_hidden_biases[lid] = used ? float(pool_hidden_biases[lid]) : 0.0f;
  }
  for (uint index = lid; index < POOL_HIDDEN_MAX * POOL_HIDDEN_MAX; index += NODES_THREADS) {
    const uint unit = index / POOL_HIDDEN_MAX;
    const uint inner = index % POOL_HIDDEN_MAX;
    reinterpret_cast<threadgroup float*>(memory.pool_hidden)[index] =
        unit < pool_hidden && inner < pool_hidden ? float(pool_hidden_weights[unit * pool_hidden + inner]) : 0.0f;
  }
  if (lid < model_dim) {
    memory.node[lid] = float(draft[depth * model_dim + lid]);
  }

  const float log_logit =
      logit - head_log_sum_exp<NODES_SIMDGROUPS>(half_index == 0u ? logit : -INFINITY, memory.reduce, context);
  if (valid) {
    scores[(2 * slot + half_index) * candidates + candidate] = half_index == 0u ? log_logit : noise;
  }
  const float floored = max(log_logit, HEAD_LOG_PROB_FLOOR);
  const float weight = valid ? exp(log_logit) : 0.0f;
  const float4 inputs = float4(log_draft, floored - log_draft, noise, 1.0f);
  float4 hidden[POOL_HIDDEN_MAX / 4u];
  METAL_PRAGMA_UNROLL
  for (uint unit = 0; unit < POOL_HIDDEN_MAX; ++unit) {
    const float value = dot(memory.pool_entry[unit], inputs);
    hidden[unit / 4u][unit % 4u] = unit < pool_hidden ? activate_silu(value) : 0.0f;
  }
  float means[NODES_UNITS];
  float peaks[NODES_UNITS];
  METAL_PRAGMA_UNROLL
  for (uint offset = 0; offset < NODES_UNITS; ++offset) {
    const uint unit = half_index * NODES_UNITS + offset;
    float value = memory.pool_hidden_biases[unit];
    METAL_PRAGMA_UNROLL
    for (uint quad = 0; quad < POOL_HIDDEN_MAX / 4u; ++quad) {
      value += dot(memory.pool_hidden[unit * (POOL_HIDDEN_MAX / 4u) + quad], hidden[quad]);
    }
    means[offset] = weight * value;
    peaks[offset] = valid ? value : -INFINITY;
  }
  const uint lane = context.simd_lane_id;
  const float mean = head_reduce<false, NODES_UNITS>(means, lane);
  const float peak = head_reduce<true, NODES_UNITS>(peaks, lane);
  const uint offset = (lane / (32u / NODES_UNITS)) % NODES_UNITS;
  if (lane % (32u / NODES_UNITS) == 0u && half_index * NODES_UNITS + offset < pool_hidden) {
    memory.reduce[(2 * context.simdgroup_index) * NODES_UNITS + offset] = mean;
    memory.reduce[(2 * context.simdgroup_index + 1) * NODES_UNITS + offset] = peak;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (context.simdgroup_index < 2u * pool_hidden) {
    const bool is_mean = context.simdgroup_index < pool_hidden;
    const uint unit = context.simdgroup_index % pool_hidden;
    const uint groups = CANDIDATES_MAX / 32u;
    const uint group = unit / NODES_UNITS * groups + lane;
    const float part = lane < groups
                           ? memory.reduce[(2 * group + (is_mean ? 0u : 1u)) * NODES_UNITS + unit % NODES_UNITS]
                           : (is_mean ? 0.0f : -INFINITY);
    const float value = is_mean ? simd_sum(part) : simd_max(part);
    if (lane == 0u) {
      memory.summary[context.simdgroup_index] = value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  head_dense(
      pool_out_weights,
      pool_out_biases,
      memory.summary,
      0,
      memory.node,
      0,
      2u * pool_hidden,
      model_dim,
      1,
      false,
      true,
      context
  );
  if (lid < model_dim) {
    nodes[slot * model_dim + lid] = bfloat(memory.node[lid]);
    node_inputs[slot * model_dim + lid] = bfloat(memory.node[lid]);
  }
}

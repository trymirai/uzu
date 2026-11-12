import definitions;
import common.activation;
import common.workgroup_reduction;
import confidence_head.head;
import generated.activation_type;
import generated.confidence_head;
import generated.weaver;

static const uint NODES_THREADS = 2 * CANDIDATES_MAX;
static const uint NODES_UNITS = POOL_HIDDEN_MAX / 2;

groupshared float4 nodes_pool_entry[POOL_HIDDEN_MAX];
groupshared float nodes_pool_hidden[POOL_HIDDEN_MAX * POOL_HIDDEN_MAX];
groupshared float nodes_pool_hidden_biases[POOL_HIDDEN_MAX];
groupshared float nodes_summary[2 * POOL_HIDDEN_MAX];

[[Kernel]] [[Public]]
[[Precondition("new", "model_dim <= crate::backends::common::gpu_types::confidence_head::MODEL_MAX && pool_hidden <= crate::backends::common::gpu_types::confidence_head::POOL_HIDDEN_MAX")]]
[[Precondition("encode", "params.candidates <= crate::backends::common::gpu_types::weaver::CANDIDATES_MAX")]]
func ConfidenceHeadNodes(
  Ptr<bf16, Access.Read> residual_logits,
  Ptr<float, Access.Read> candidate_logits,
  Ptr<float, Access.Read> pool_features,
  Ptr<uint, Access.Read> round_metadata,
  Ptr<uint, Access.Read> round_valid,
  Ptr<bf16, Access.Read> draft,
  Ptr<bf16, Access.Read> pool_entry_weights,
  Ptr<bf16, Access.Read> pool_entry_biases,
  Ptr<bf16, Access.Read> pool_hidden_weights,
  Ptr<bf16, Access.Read> pool_hidden_biases,
  Ptr<bf16, Access.Read> pool_out_weights,
  Ptr<bf16, Access.Read> pool_out_biases,
  Ptr<float, Access.ReadWrite> scores,
  Ptr<bf16, Access.ReadWrite> nodes,
  Ptr<bf16, Access.ReadWrite> node_inputs,
  ConfidenceHeadParams params,
  uint rows,
  [[Groups("rows")]] uint row,
  [[Threads("1024")]] uint lid,
  [[Specialize]] uint model_dim,
  [[Specialize]] uint pool_hidden,
) {
  let slot = round_metadata[uint(MetadataIdx.TreeSlot) * rows + row];
  if (round_valid[row] == 0 || slot >= params.tree_slot_count) {
    return;
  }
  let depth = round_metadata[uint(MetadataIdx.Depth) * rows + row];
  let candidates = params.candidates;
  let features = pool_features + depth * POOL_FEATURES * candidates;
  let invocation = subgroup_invocation();
  let candidate = invocation % CANDIDATES_MAX;
  let part = invocation / CANDIDATES_MAX;
  let valid = candidate < candidates;
  let logit = valid
    ? candidate_logits[depth * candidates + candidate] + residual_logits[row * candidates + candidate].toFloat()
    : NEGATIVE_INFINITY;
  let log_draft = valid ? features[candidate] : 0.0;
  let noise = valid ? features[candidates + candidate] : 0.0;
  if (invocation < POOL_HIDDEN_MAX) {
    let used = invocation < pool_hidden;
    let entry = pool_entry_weights + POOL_INPUTS * invocation;
    nodes_pool_entry[invocation] = used
      ? float4(entry[0].toFloat(), entry[1].toFloat(), entry[2].toFloat(), pool_entry_biases[invocation].toFloat())
      : float4(0.0);
    nodes_pool_hidden_biases[invocation] = used ? pool_hidden_biases[invocation].toFloat() : 0.0;
  }
  for (uint index = invocation; index < POOL_HIDDEN_MAX * POOL_HIDDEN_MAX; index += NODES_THREADS) {
    let unit = index / POOL_HIDDEN_MAX;
    let inner = index % POOL_HIDDEN_MAX;
    nodes_pool_hidden[index] =
      unit < pool_hidden && inner < pool_hidden ? pool_hidden_weights[unit * pool_hidden + inner].toFloat() : 0.0;
  }

  let log_logit = logit - head_log_sum_exp<NODES_THREADS>(part == 0 ? logit : NEGATIVE_INFINITY);
  if (valid) {
    scores[(2 * slot + part) * candidates + candidate] = part == 0 ? log_logit : noise;
  }
  let weight = valid ? exp(log_logit) : 0.0;
  let inputs = float4(log_draft, max(log_logit, HEAD_LOG_PROB_FLOOR) - log_draft, noise, 1.0);
  float hidden[POOL_HIDDEN_MAX];
  for (uint unit = 0; unit < POOL_HIDDEN_MAX; unit++) {
    hidden[unit] = unit < pool_hidden ? activate(dot(nodes_pool_entry[unit], inputs), ActivationType.SILU) : 0.0;
  }
  for (uint offset = 0; offset < NODES_UNITS; offset++) {
    let unit = part * NODES_UNITS + offset;
    var value = nodes_pool_hidden_biases[unit];
    for (uint inner = 0; inner < POOL_HIDDEN_MAX; inner++) {
      value += nodes_pool_hidden[unit * POOL_HIDDEN_MAX + inner] * hidden[inner];
    }
    let mean = segment_reduce<Sum<float>, NODES_THREADS>(weight * value, CANDIDATES_MAX);
    let peak = segment_reduce<Max<float>, NODES_THREADS>(valid ? value : NEGATIVE_INFINITY, CANDIDATES_MAX);
    if (candidate == 0 && unit < pool_hidden) {
      nodes_summary[unit] = mean;
      nodes_summary[pool_hidden + unit] = peak;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  if (invocation < model_dim) {
    let summary = 2 * pool_hidden;
    var value = pool_out_biases[invocation].toFloat();
    for (uint index = 0; index < summary; index++) {
      value += pool_out_weights[invocation * summary + index].toFloat() * nodes_summary[index];
    }
    let node = bf16.fromFloat(value + draft[depth * model_dim + invocation].toFloat());
    nodes[slot * model_dim + invocation] = node;
    node_inputs[slot * model_dim + invocation] = node;
  }
}

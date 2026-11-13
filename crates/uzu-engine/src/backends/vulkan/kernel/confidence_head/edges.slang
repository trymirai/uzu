import definitions;
import common.activation;
import common.ranking;
import common.workgroup_reduction;
import confidence_head.head;
import generated.activation_type;
import generated.confidence_head;
import generated.weaver;

static const uint EDGES_THREADS = CANDIDATES_MAX;
static const uint EDGES_INPUT_MAX = 3 * MODEL_MAX + SLOT_NUMBERS;

static const uint EDGES_INPUT = 0;
static const uint EDGES_STATE = EDGES_INPUT + CHILDREN_MAX * EDGES_INPUT_MAX;
static const uint EDGES_BLOCK = EDGES_STATE + CHILDREN_MAX * MODEL_MAX;
static const uint EDGES_RAISED = EDGES_BLOCK + CHILDREN_MAX * MODEL_MAX;
static const uint EDGES_TOKENS = EDGES_RAISED + CHILDREN_MAX * EDGE_HIDDEN_MAX;
static const uint EDGES_NUMBERS = EDGES_TOKENS + CHILDREN_MAX * MODEL_MAX;
static const uint EDGES_DELTAS = EDGES_NUMBERS + CHILDREN_MAX * SLOT_NUMBERS;
static const uint EDGES_EDGES = EDGES_DELTAS + CHILDREN_MAX;
static const uint EDGES_NODE = EDGES_EDGES + CHILDREN_MAX;
static const uint EDGES_COEFFICIENTS = EDGES_NODE + MODEL_MAX;

groupshared float edges_memory[EDGES_COEFFICIENTS + 2];
groupshared uint edges_pools[CHILDREN_MAX];

void edges_dense(
  Ptr<bf16, Access.Read> weights,
  Ptr<bf16, Access.Read> biases,
  uint input,
  uint input_stride,
  uint output,
  uint output_stride,
  uint inputs,
  uint outputs,
  uint rows,
  bool silu,
  bool accumulate,
) {
  var lanes = 1u;
  while (lanes < SIMD_SIZE && 2 * lanes * outputs <= EDGES_THREADS) {
    lanes *= 2;
  }
  let invocation = subgroup_invocation();
  let lane = invocation % lanes;
  for (uint first = 0; first < outputs; first += EDGES_THREADS / lanes) {
    let unit = first + invocation / lanes;
    float sums[CHILDREN_MAX];
    for (uint row = 0; row < CHILDREN_MAX; row++) {
      sums[row] = 0.0;
    }
    if (unit < outputs) {
      for (uint column = lane; column < inputs; column += lanes) {
        let weight = weights[unit * inputs + column].toFloat();
        for (uint row = 0; row < rows; row++) {
          sums[row] = fma(weight, edges_memory[input + row * input_stride + column], sums[row]);
        }
      }
    }
    for (uint row = 0; row < CHILDREN_MAX; row++) {
      sums[row] = segment_reduce<Sum<float>, EDGES_THREADS>(sums[row], lanes);
    }
    if (lane == 0 && unit < outputs) {
      for (uint row = 0; row < rows; row++) {
        var value = sums[row] + biases[unit].toFloat();
        if (silu) {
          value = activate(value, ActivationType.SILU);
        }
        let index = output + row * output_stride + unit;
        edges_memory[index] = accumulate ? edges_memory[index] + value : value;
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
}

void edges_layer_norm(
  uint input,
  uint output,
  uint stride,
  uint count,
  uint rows,
  Ptr<bf16, Access.Read> scales,
  Ptr<bf16, Access.Read> biases,
  float epsilon,
) {
  let invocation = subgroup_invocation();
  let row = invocation / SIMD_SIZE;
  let lane = invocation % SIMD_SIZE;
  let active = row < rows;
  var total = 0.0;
  if (active) {
    for (uint index = lane; index < count; index += SIMD_SIZE) {
      total += edges_memory[input + row * stride + index];
    }
  }
  let mean = segment_reduce<Sum<float>, EDGES_THREADS>(total, SIMD_SIZE) / float(count);
  var spread = 0.0;
  if (active) {
    for (uint index = lane; index < count; index += SIMD_SIZE) {
      let centered = edges_memory[input + row * stride + index] - mean;
      spread += centered * centered;
    }
  }
  let scale = rsqrt(segment_reduce<Sum<float>, EDGES_THREADS>(spread, SIMD_SIZE) / float(count) + epsilon);
  if (active) {
    for (uint index = lane; index < count; index += SIMD_SIZE) {
      let value = edges_memory[input + row * stride + index];
      edges_memory[output + row * stride + index] =
        (value - mean) * scale * scales[index].toFloat() + biases[index].toFloat();
    }
  }
  GroupMemoryBarrierWithGroupSync();
}

[[Kernel]] [[Public]]
func ConfidenceHeadEdges(
  [[Optional("!refine")]] Ptr<uint, Access.Read> node_metadata,
  [[Optional("!refine")]] Ptr<uint, Access.Read> node_valid,
  [[Optional("!refine")]] Ptr<uint, Access.Read> candidate_ids,
  [[Optional("!refine")]] Ptr<float, Access.Read> pool_features,
  [[Optional("!refine")]] Ptr<bf16, Access.Read> slot_stats,
  Ptr<uint, Access.ReadWrite> slot_rows,
  [[Optional("!refine")]] Ptr<uint, Access.ReadWrite> row_slots,
  Ptr<bf16, Access.Read> nodes,
  [[Optional("refine")]] Ptr<bf16, Access.Read> node_shifts,
  [[Optional("refine")]] Ptr<bf16, Access.Read> child_shifts,
  Ptr<bf16, Access.Read> token_features,
  Ptr<float, Access.ReadWrite> child_numbers,
  Ptr<uint, Access.ReadWrite> child_pools,
  Ptr<float, Access.Read> scores,
  Ptr<uint, Access.ReadWrite> frontier,
  Ptr<uint, Access.ReadWrite> packed_tree,
  [[Optional("!refine")]] Ptr<bf16, Access.ReadWrite> child_inputs,
  Ptr<bf16, Access.Read> input_norm,
  Ptr<bf16, Access.Read> entry_weights,
  Ptr<bf16, Access.Read> entry_biases,
  Ptr<bf16, Access.Read> block_norm,
  Ptr<bf16, Access.Read> up_weights,
  Ptr<bf16, Access.Read> up_biases,
  Ptr<bf16, Access.Read> down_weights,
  Ptr<bf16, Access.Read> down_biases,
  Ptr<bf16, Access.Read> final_norm,
  Ptr<bf16, Access.Read> readout_weights,
  Ptr<bf16, Access.Read> readout_biases,
  ConfidenceHeadParams params,
  uint groups,
  [[Groups("groups")]] uint group,
  [[Threads("512")]] uint lid,
  [[Specialize]] uint model_dim,
  [[Specialize]] uint edge_hidden,
  [[Specialize]] bool refine,
) {
  let tree_slot_count = params.tree_slot_count;
  let children = params.children;
  let candidates = params.candidates;
  let capacity = tree_slot_count * children;
  let slot = refine ? group : node_metadata[uint(MetadataIdx.TreeSlot) * groups + group];
  if (slot >= tree_slot_count || (refine ? slot_rows[slot] == HEAD_NONE : node_valid[group] == 0)) {
    return;
  }
  let first_entry = slot * children;
  let invocation = subgroup_invocation();

  if (invocation < children) {
    edges_pools[invocation] = child_pools[first_entry + invocation];
  }
  if (invocation < children * SLOT_NUMBERS) {
    edges_memory[EDGES_NUMBERS + invocation] = child_numbers[first_entry * SLOT_NUMBERS + invocation];
  }
  GroupMemoryBarrierWithGroupSync();
  if (!refine) {
    if (invocation < candidates) {
      let depth = node_metadata[uint(MetadataIdx.Depth) * groups + group];
      let token = candidate_ids[depth * candidates + invocation];
      let features = pool_features + depth * POOL_FEATURES * candidates;
      for (uint child = 0; child < children; child++) {
        let entry = first_entry + child;
        if (frontier[uint(FrontierIdx.TokenId) * capacity + entry] == token) {
          float numbers[SLOT_NUMBERS] = {
            features[invocation],
            max(scores[2 * slot * candidates + invocation], HEAD_LOG_PROB_FLOOR),
            scores[(2 * slot + 1) * candidates + invocation],
            features[2 * candidates + invocation]
          };
          edges_pools[child] = invocation;
          child_pools[entry] = invocation;
          for (uint number = 0; number < SLOT_NUMBERS; number++) {
            let value = (numbers[number] - slot_stats[number].toFloat()) / slot_stats[SLOT_NUMBERS + number].toFloat();
            edges_memory[EDGES_NUMBERS + child * SLOT_NUMBERS + number] = value;
            child_numbers[entry * SLOT_NUMBERS + number] = value;
          }
        }
      }
    }
    if (invocation == 0) {
      var own_row = 0u;
      if (slot != 0) {
        let parent = packed_tree[uint(TreeIdx.ParentSlot) * tree_slot_count + slot];
        let own_token = packed_tree[uint(TreeIdx.TokenId) * tree_slot_count + slot];
        for (uint child = 0; child < children; child++) {
          if (frontier[uint(FrontierIdx.TokenId) * capacity + parent * children + child] == own_token) {
            own_row = 1 + parent * children + child;
          }
        }
      }
      slot_rows[slot] = own_row;
      row_slots[own_row] = slot;
    }
    GroupMemoryBarrierWithGroupSync();
  }

  let row = refine ? slot_rows[slot] : 0u;
  let inputs = 3 * model_dim + SLOT_NUMBERS;
  let own_child = invocation / SIMD_SIZE;
  let lane = invocation % SIMD_SIZE;
  if (own_child < children) {
    let entry = first_entry + own_child;
    let token = frontier[uint(FrontierIdx.TokenId) * capacity + entry];
    let line = EDGES_INPUT + own_child * EDGES_INPUT_MAX;
    for (uint unit = lane; unit < model_dim; unit += SIMD_SIZE) {
      let node = nodes[slot * model_dim + unit].toFloat() + (refine ? node_shifts[row * model_dim + unit].toFloat() : 0.0);
      let vector = token_features[token * model_dim + unit].toFloat()
        + (refine ? child_shifts[(1 + entry) * model_dim + unit].toFloat() : 0.0);
      edges_memory[line + unit] = node;
      edges_memory[line + model_dim + unit] = vector;
      edges_memory[line + 2 * model_dim + unit] = node * vector;
      edges_memory[EDGES_TOKENS + own_child * MODEL_MAX + unit] = vector;
      if (own_child == 0) {
        edges_memory[EDGES_NODE + unit] = node;
      }
    }
    if (lane < SLOT_NUMBERS) {
      edges_memory[line + 3 * model_dim + lane] = edges_memory[EDGES_NUMBERS + own_child * SLOT_NUMBERS + lane];
    }
  }
  GroupMemoryBarrierWithGroupSync();

  let epsilon = params.epsilon;
  edges_layer_norm(EDGES_INPUT, EDGES_INPUT, EDGES_INPUT_MAX, inputs, children, input_norm, input_norm + inputs, epsilon);
  edges_dense(entry_weights, entry_biases, EDGES_INPUT, EDGES_INPUT_MAX, EDGES_STATE, MODEL_MAX, inputs, model_dim,
    children, false, false);
  edges_layer_norm(EDGES_STATE, EDGES_BLOCK, MODEL_MAX, model_dim, children, block_norm, block_norm + model_dim, epsilon);
  edges_dense(up_weights, up_biases, EDGES_BLOCK, MODEL_MAX, EDGES_RAISED, EDGE_HIDDEN_MAX, model_dim, edge_hidden,
    children, true, false);
  edges_dense(down_weights, down_biases, EDGES_RAISED, EDGE_HIDDEN_MAX, EDGES_STATE, MODEL_MAX, edge_hidden, model_dim,
    children, false, true);
  edges_layer_norm(EDGES_STATE, EDGES_BLOCK, MODEL_MAX, model_dim, children, final_norm, final_norm + model_dim, epsilon);
  edges_dense(readout_weights, readout_biases, EDGES_BLOCK, MODEL_MAX, EDGES_DELTAS, 1, model_dim, 1, children, false,
    false);
  edges_dense(readout_weights + model_dim, readout_biases + 1, EDGES_NODE, 0, EDGES_COEFFICIENTS, 0, model_dim, 2, 1,
    false, false);

  let alpha = min(activate(edges_memory[EDGES_COEFFICIENTS], ActivationType.SOFTPLUS), params.logit_cap);
  let beta = min(activate(edges_memory[EDGES_COEFFICIENTS + 1], ActivationType.SOFTPLUS), params.logit_cap);
  var score = invocation < candidates
    ? alpha * scores[2 * slot * candidates + invocation] + beta * scores[(2 * slot + 1) * candidates + invocation]
    : NEGATIVE_INFINITY;
  for (uint child = 0; child < children; child++) {
    score += edges_pools[child] == invocation ? edges_memory[EDGES_DELTAS + child] : 0.0;
  }
  let log_sum = head_log_sum_exp<EDGES_THREADS>(score);
  let parent_path = asfloat(packed_tree[uint(TreeIdx.PathLogprobBits) * tree_slot_count + slot]);
  for (uint child = 0; child < children; child++) {
    if (edges_pools[child] == invocation) {
      let entry = first_entry + child;
      let edge = min(score - log_sum, 0.0);
      edges_memory[EDGES_EDGES + child] = edge;
      frontier[uint(FrontierIdx.EdgeLogprobBits) * capacity + entry] = asuint(edge);
      if (!refine) {
        frontier[uint(FrontierIdx.PathLogprobBits) * capacity + entry] = asuint(parent_path + edge);
        frontier[uint(FrontierIdx.PathScoreKey) * capacity + entry] = float_order_key(asuint(parent_path + edge));
      }
    }
  }
  if (!refine) {
    let width = model_dim + SLOT_NUMBERS;
    for (uint index = invocation; index < children * width; index += EDGES_THREADS) {
      let child = index / width;
      let column = index % width;
      let value = column < model_dim
        ? edges_memory[EDGES_TOKENS + child * MODEL_MAX + column]
        : edges_memory[EDGES_NUMBERS + child * SLOT_NUMBERS + column - model_dim];
      child_inputs[(first_entry + child) * width + column] = bf16.fromFloat(value);
    }
    return;
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint other = invocation; other < tree_slot_count; other += EDGES_THREADS) {
    if (packed_tree[uint(TreeIdx.ParentSlot) * tree_slot_count + other] == slot) {
      let token = packed_tree[uint(TreeIdx.TokenId) * tree_slot_count + other];
      for (uint child = 0; child < children; child++) {
        if (frontier[uint(FrontierIdx.TokenId) * capacity + first_entry + child] == token) {
          packed_tree[uint(TreeIdx.EdgeLogprobBits) * tree_slot_count + other] =
            asuint(edges_memory[EDGES_EDGES + child]);
        }
      }
    }
  }
}

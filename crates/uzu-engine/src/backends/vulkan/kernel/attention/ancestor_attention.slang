import definitions;
import attention.attention;
import generated.kv_layout;
import generated.weaver;

static const uint ANCESTOR_ATTENTION_QUERY = 0;
static const uint ANCESTOR_ATTENTION_KEY = 1;
static const uint ANCESTOR_ATTENTION_VALUE = 2;

// The row's own query, key and value codes, the value's centered, and their scales.
groupshared float ancestor_codes[3 * ATTENTION_DIMS * SIMD_SIZE];
groupshared float ancestor_scales[3];

// One row of one head: the prefix keys, the row's ancestor slots, then its own key, each [.., num_heads, HEAD_DIM] of
// INT8 codes with a scale per row and head.
struct AncestorKeys : IAttentionKeys {
  Ptr<int8_t, Access.Read> prefix_k_codes;
  Ptr<bf16, Access.Read> prefix_k_scales;
  Ptr<uint8_t, Access.Read> prefix_v_codes;
  Ptr<bf16, Access.Read> prefix_v_scales;
  Ptr<int8_t, Access.ReadWrite> node_k_codes;
  Ptr<bf16, Access.ReadWrite> node_k_scales;
  Ptr<uint8_t, Access.ReadWrite> node_v_codes;
  Ptr<bf16, Access.ReadWrite> node_v_scales;
  Ptr<uint, Access.Read> ancestor_indices;
  uint ancestors;
  uint ancestor_count;
  uint prefix_length;
  uint num_heads;
  uint head;
  uint head_dim;

  // The head's row of key `key` in its codes: a prefix row below prefix_length, then the listed ancestors' slots.
  uint row(uint key) {
    if (key < prefix_length) {
      return key * num_heads + head;
    }
    return ancestor_indices[ancestors + key - prefix_length] * num_heads + head;
  }

  bool own(uint key) {
    return key >= prefix_length + ancestor_count;
  }

  float query(uint j) {
    return ancestor_codes[ANCESTOR_ATTENTION_QUERY * head_dim + j];
  }

  float query_scale() {
    return ancestor_scales[ANCESTOR_ATTENTION_QUERY];
  }

  bool used(uint key) {
    return true;
  }

  float key(uint key, uint j) {
    if (own(key)) {
      return ancestor_codes[ANCESTOR_ATTENTION_KEY * head_dim + j];
    }
    let element = row(key) * head_dim + j;
    if (key < prefix_length) {
      return float(prefix_k_codes[element]);
    }
    return float(node_k_codes[element]);
  }

  float key_scale(uint key) {
    if (own(key)) {
      return ancestor_scales[ANCESTOR_ATTENTION_KEY];
    }
    if (key < prefix_length) {
      return prefix_k_scales[row(key)].toFloat();
    }
    return node_k_scales[row(key)].toFloat();
  }

  float value(uint key, uint j) {
    if (own(key)) {
      return ancestor_codes[ANCESTOR_ATTENTION_VALUE * head_dim + j];
    }
    let element = row(key) * head_dim + j;
    if (key < prefix_length) {
      return float(int(prefix_v_codes[element]) - int(V_CODE_BIAS));
    }
    return float(int(node_v_codes[element]) - int(V_CODE_BIAS));
  }

  float value_scale(uint key) {
    if (own(key)) {
      return ancestor_scales[ANCESTOR_ATTENTION_VALUE];
    }
    if (key < prefix_length) {
      return prefix_v_scales[row(key)].toFloat();
    }
    return node_v_scales[row(key)].toFloat();
  }
}

// Each row of `current_qkv` [rows, 3, num_heads, HEAD_DIM] attends the prefix, its first ancestor_counts[row] slots of
// ancestor_indices [rows, ancestor_stride] in the node codes, and itself: its query and key rotated by RoPE at depth + 1
// and rounded to BF16, then, with its value, quantized as AttentionPrepare does. With node_capacity > 0 its key and value
// codes then enter slot node_indices[row]. Unchecked: depths below max_depth, counts and slots in range, destinations
// distinct and disjoint from every listed slot.
[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_DIM == 128 && backend_opaque_layout == 0")]]
func AncestorAttention(
  Ptr<int8_t, Access.Read> prefix_k_codes,
  Ptr<bf16, Access.Read> prefix_k_scales,
  Ptr<uint8_t, Access.Read> prefix_v_codes,
  Ptr<bf16, Access.Read> prefix_v_scales,
  Ptr<int8_t, Access.ReadWrite> node_k_codes,
  Ptr<bf16, Access.ReadWrite> node_k_scales,
  Ptr<uint8_t, Access.ReadWrite> node_v_codes,
  Ptr<bf16, Access.ReadWrite> node_v_scales,
  Ptr<bf16, Access.Read> current_qkv,
  Ptr<float, Access.Read> cosines,
  Ptr<float, Access.Read> sines,
  Ptr<uint, Access.Read> node_metadata,
  Ptr<uint, Access.Read> ancestor_indices,
  Ptr<uint, Access.Read> ancestor_counts,
  Ptr<uint, Access.Read> node_indices,
  Ptr<bf16, Access.ReadWrite> output,
  uint rows,
  uint prefix_length,
  uint ancestor_stride,
  uint node_capacity,
  uint max_depth,
  float scale,
  [[Groups("num_heads")]] uint head,
  [[Groups("rows")]] uint row,
  [[Threads("1024")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] uint num_heads,
  [[Specialize]] uint backend_opaque_layout,
) {
  static_assert(ATTENTION_THREADS == 1024, "one workgroup of Threads(1024) per row and head");
  let invocation = subgroup_invocation();
  let group = invocation / SIMD_SIZE;
  let lane = invocation % SIMD_SIZE;
  let dims = HEAD_DIM / SIMD_SIZE;
  let first = lane * dims;
  let model_dim = num_heads * HEAD_DIM;
  let qkv = row * 3 * model_dim + head * HEAD_DIM;
  let table = (node_metadata[uint(MetadataIdx.Depth) * rows + row] + 1) * HEAD_DIM;
  let half_dim = HEAD_DIM / 2;

  // Every group quantizes the row's own query, key and value alike; group 0 shares them.
  for (uint component = 0; component < 3; component++) {
    let base = qkv + component * model_dim;
    float values[ATTENTION_DIMS];
    for (uint a = 0; a < dims; a++) {
      let channel = first + a;
      var value = current_qkv[base + channel].toFloat();
      if (component != ANCESTOR_ATTENTION_VALUE) {
        let low = channel < half_dim;
        let paired = current_qkv[base + (low ? channel + half_dim : channel - half_dim)].toFloat();
        value = bf16.fromFloat(rope_rotated(value, paired, low, cosines[table + channel], sines[table + channel]))
          .toFloat();
      }
      values[a] = value;
    }
    attention_rotate<ATTENTION_THREADS>(values, dims, lane);
    int codes[ATTENTION_DIMS];
    let code_scale = attention_quantize<ATTENTION_THREADS>(values, dims, codes);
    if (group == 0) {
      for (uint a = 0; a < dims; a++) {
        ancestor_codes[component * HEAD_DIM + first + a] = float(codes[a]);
      }
      if (lane == 0) {
        ancestor_scales[component] = code_scale.toFloat();
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();

  AncestorKeys source;
  source.prefix_k_codes = prefix_k_codes;
  source.prefix_k_scales = prefix_k_scales;
  source.prefix_v_codes = prefix_v_codes;
  source.prefix_v_scales = prefix_v_scales;
  source.node_k_codes = node_k_codes;
  source.node_k_scales = node_k_scales;
  source.node_v_codes = node_v_codes;
  source.node_v_scales = node_v_scales;
  source.ancestor_indices = ancestor_indices;
  source.ancestors = row * ancestor_stride;
  source.ancestor_count = ancestor_counts[row];
  source.prefix_length = prefix_length;
  source.num_heads = num_heads;
  source.head = head;
  source.head_dim = HEAD_DIM;
  let sequence_length = prefix_length + source.ancestor_count + 1;
  let statistics = attention_single_query(source, HEAD_DIM, sequence_length, scale);

  float values[ATTENTION_DIMS];
  for (uint a = 0; a < dims; a++) {
    values[a] = attention_numerator(first + a) / statistics.y;
  }
  attention_unrotate<ATTENTION_THREADS>(values, dims, lane);
  if (group == 0) {
    for (uint a = 0; a < dims; a++) {
      output[(row * num_heads + head) * HEAD_DIM + first + a] = bf16.fromFloat(values[a]);
    }
  }
  // The row's key and value enter its slot, which no row of this dispatch reads.
  if (node_capacity > 0 && invocation < HEAD_DIM) {
    let node_row = node_indices[row] * num_heads + head;
    let element = node_row * HEAD_DIM + invocation;
    node_k_codes[element] = int8_t(int(ancestor_codes[ANCESTOR_ATTENTION_KEY * HEAD_DIM + invocation]));
    node_v_codes[element] =
      uint8_t(int(ancestor_codes[ANCESTOR_ATTENTION_VALUE * HEAD_DIM + invocation]) + int(V_CODE_BIAS));
    if (invocation == 0) {
      node_k_scales[node_row] = bf16.fromFloat(ancestor_scales[ANCESTOR_ATTENTION_KEY]);
      node_v_scales[node_row] = bf16.fromFloat(ancestor_scales[ANCESTOR_ATTENTION_VALUE]);
    }
  }
}

import definitions;
import attention.mask;
import common.workgroup_reduction;
import generated.attention_gemm;
import generated.kv_layout;
import generated.ring;
import generated.trie;

// Metal's AttentionGemm on scalar tiles of INT8 codes: a workgroup takes ATTENTION_GEMM_QUERIES of a key-value head's
// gqa_factor * suffix_length query rows, which AttentionPrepare stores in turn, and walks their keys in blocks of
// ATTENTION_GEMM_BLOCK_KEYS, each staged once for the group's query heads: key codes, then value codes as FP32 in
// halves of GEMM_VALUE_KEYS, with their scales. The run of GEMM_ROW_LANES (L) invocations at (subgroup order) L r holds
// row r: key column `lane` + L j of a block's scores, as integer dots, output dimensions 4 (lane + L m) to
// 4 (lane + L m) + 3, and the row's base-2 online softmax. Split s of `blocks` takes key blocks s, s + blocks, ... and
// writes AttentionTwoPass1's partial (head * suffix_length + query) * blocks + s, -inf without an admitted key.

static const uint GEMM_THREADS = 256;
static const uint GEMM_ROW_LANES = 8;
static const uint GEMM_MAX_HEAD_DIM = 256;
static const uint GEMM_VALUE_KEYS = ATTENTION_GEMM_BLOCK_KEYS / 2;
// Rows of codes in 16-byte quads, padded by one.
static const uint GEMM_MAX_ROW_QUADS = GEMM_MAX_HEAD_DIM / 16 + 1;
static const float GEMM_LOG2E = 1.4426950408889634;

groupshared uint4 attention_gemm_queries[ATTENTION_GEMM_QUERIES * GEMM_MAX_ROW_QUADS];
// Key code rows of head_dim / 16 + 1 quads, or value rows of head_dim / 4.
groupshared uint4 attention_gemm_block[GEMM_VALUE_KEYS * GEMM_MAX_HEAD_DIM / 4];
groupshared float attention_gemm_weights[ATTENTION_GEMM_QUERIES * ATTENTION_GEMM_BLOCK_KEYS];
groupshared float attention_gemm_scales[ATTENTION_GEMM_BLOCK_KEYS];

// Rows [first_row, first_row + rows) of head `head` of a buffer of `num_kv_heads` heads into the block tile, key codes
// as they are and value codes less V_CODE_BIAS as FP32, and their scales; rows at or past `valid_rows` are zero.
void attention_gemm_stage<T>(
  Ptr<T, Access.Read> codes,
  Ptr<bf16, Access.Read> scales,
  uint head,
  uint num_kv_heads,
  uint first_row,
  uint rows,
  uint valid_rows,
  uint head_dim,
  uint invocation,
  bool values
) {
  let quads = head_dim / 16;
  for (uint chunk = invocation; chunk < rows * quads; chunk += GEMM_THREADS) {
    let row = chunk / quads;
    let column = chunk % quads;
    var words = uint4(0);
    if (first_row + row < valid_rows) {
      words = load_aligned<uint4, 16>(codes, ((first_row + row) * num_kv_heads + head) * head_dim + 16 * column);
    }
    if (values) {
      [unroll]
      for (uint word = 0; word < 4; word++) {
        let centered = (int4(words[word] >> uint4(0, 8, 16, 24)) & 0xff) - int(V_CODE_BIAS);
        attention_gemm_block[row * (head_dim / 4) + 4 * column + word] = asuint(float4(centered));
      }
    } else {
      attention_gemm_block[row * (quads + 1) + column] = words;
    }
  }
  if (invocation < rows) {
    var scale = 0.0;
    if (first_row + invocation < valid_rows) {
      scale = scales[(first_row + invocation) * num_kv_heads + head].toFloat();
    }
    attention_gemm_scales[invocation] = scale;
  }
}

[[Kernel]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256")]]
[[Precondition("encode", "gqa_factor > 0 && num_heads == gqa_factor * num_kv_heads")]]
[[Precondition("encode", "suffix_length <= sequence_length")]]
func AttentionGemm(
  Ptr<int8_t, Access.Read> q_codes,
  Ptr<bf16, Access.Read> q_scales,
  Ptr<int8_t, Access.Read> k_codes,
  Ptr<bf16, Access.Read> k_scales,
  Ptr<uint8_t, Access.Read> v_codes,
  Ptr<bf16, Access.Read> v_scales,
  Ptr<float, Access.ReadWrite> out,
  Ptr<float, Access.ReadWrite> sums,
  Ptr<float, Access.ReadWrite> maxs,
  uint gqa_factor,
  uint num_kv_heads,
  uint sequence_length,
  [[Optional("is_kv_cache_ring")]] RingParams ring_params,
  float scale,
  [[Optional("is_trie")]] Ptr<TrieNode, Access.Read> trie,
  [[Optional("is_sliding_window")]] uint sliding_window_size,
  uint num_heads,
  uint suffix_length,
  uint blocks,
  [[Groups("(gqa_factor * suffix_length).div_ceil(crate::backends::vulkan::gpu_types::attention_gemm::ATTENTION_GEMM_QUERIES)")]] uint tile,
  [[Groups("num_kv_heads")]] uint kv_head,
  [[Groups("blocks")]] uint split,
  [[Threads("256")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
) {
  static_assert(GEMM_THREADS == ATTENTION_GEMM_QUERIES * GEMM_ROW_LANES, "GEMM_ROW_LANES invocations a query row");
  static_assert(ATTENTION_GEMM_BLOCK_KEYS * GEMM_MAX_ROW_QUADS <= GEMM_VALUE_KEYS * GEMM_MAX_HEAD_DIM / 4, "key rows fit");
  static_assert(
    16 * (ATTENTION_GEMM_QUERIES * GEMM_MAX_ROW_QUADS + GEMM_VALUE_KEYS * GEMM_MAX_HEAD_DIM / 4) +
        4 * (ATTENTION_GEMM_QUERIES * ATTENTION_GEMM_BLOCK_KEYS + ATTENTION_GEMM_BLOCK_KEYS) ==
      ATTENTION_GEMM_SHARED_BYTES,
    "the shared memory the host checks"
  );
  let invocation = subgroup_invocation();
  let row = invocation / GEMM_ROW_LANES;
  let lane = invocation % GEMM_ROW_LANES;
  let quads = HEAD_DIM / 16;
  let row_quads = quads + 1;
  let grouped_rows = gqa_factor * suffix_length;
  let first_row = tile * ATTENTION_GEMM_QUERIES;
  let group_row = first_row + row;
  let valid_query = group_row < grouped_rows;
  let query = min(group_row, grouped_rows - 1) % suffix_length;
  let mask = AttentionMask(
    sequence_length, suffix_length, ring_params, trie, sliding_window_size, is_kv_cache_ring, is_causal, is_trie,
    is_sliding_window
  );
  let query_position = mask.query_position(query);
  // Row group_row of the group is query row kv_head * grouped_rows + group_row, head * suffix_length + query.
  let query_rows = kv_head * grouped_rows;

  for (uint chunk = invocation; chunk < ATTENTION_GEMM_QUERIES * quads; chunk += GEMM_THREADS) {
    let chunk_row = chunk / quads;
    let column = chunk % quads;
    var words = uint4(0);
    if (first_row + chunk_row < grouped_rows) {
      words = load_aligned<uint4, 16>(q_codes, (query_rows + first_row + chunk_row) * HEAD_DIM + 16 * column);
    }
    attention_gemm_queries[chunk_row * row_quads + column] = words;
  }
  // The query's scale with the softmax's, in base 2.
  var query_scale = 0.0;
  if (valid_query) {
    query_scale = scale * GEMM_LOG2E * q_scales[query_rows + group_row].toFloat();
  }

  var row_max = NEGATIVE_INFINITY;
  var row_sum = 0.0;
  float output[GEMM_MAX_HEAD_DIM / GEMM_ROW_LANES];
  [unroll]
  for (uint i = 0; i < GEMM_MAX_HEAD_DIM / GEMM_ROW_LANES; i++) {
    output[i] = 0.0;
  }

  // Under causality no row of the tile sees a key past its latest query's position.
  var key_blocks = (sequence_length + ATTENTION_GEMM_BLOCK_KEYS - 1) / ATTENTION_GEMM_BLOCK_KEYS;
  if (is_causal) {
    let last_row = min(first_row + ATTENTION_GEMM_QUERIES, grouped_rows) - 1;
    let last_query =
      first_row / suffix_length == last_row / suffix_length ? last_row % suffix_length : suffix_length - 1;
    key_blocks = min(key_blocks, (mask.prefix_length + last_query) / ATTENTION_GEMM_BLOCK_KEYS + 1);
  }
  for (uint block = split; block < key_blocks; block += blocks) {
    let first_key = block * ATTENTION_GEMM_BLOCK_KEYS;
    GroupMemoryBarrierWithGroupSync();
    attention_gemm_stage(
      k_codes, k_scales, kv_head, num_kv_heads, first_key, ATTENTION_GEMM_BLOCK_KEYS, sequence_length, HEAD_DIM,
      invocation, false
    );
    GroupMemoryBarrierWithGroupSync();

    int dots[ATTENTION_GEMM_BLOCK_KEYS / GEMM_ROW_LANES];
    [unroll]
    for (uint j = 0; j < ATTENTION_GEMM_BLOCK_KEYS / GEMM_ROW_LANES; j++) {
      dots[j] = 0;
    }
    for (uint quad = 0; quad < quads; quad++) {
      let q = attention_gemm_queries[row * row_quads + quad];
      [unroll]
      for (uint j = 0; j < ATTENTION_GEMM_BLOCK_KEYS / GEMM_ROW_LANES; j++) {
        let k = attention_gemm_block[(lane + GEMM_ROW_LANES * j) * row_quads + quad];
        dots[j] += dot_i8x4(q.x, k.x) + dot_i8x4(q.y, k.y) + dot_i8x4(q.z, k.z) + dot_i8x4(q.w, k.w);
      }
    }

    float scores[ATTENTION_GEMM_BLOCK_KEYS / GEMM_ROW_LANES];
    bool used[ATTENTION_GEMM_BLOCK_KEYS / GEMM_ROW_LANES];
    var block_max = NEGATIVE_INFINITY;
    [unroll]
    for (uint j = 0; j < ATTENTION_GEMM_BLOCK_KEYS / GEMM_ROW_LANES; j++) {
      let column = lane + GEMM_ROW_LANES * j;
      let key = first_key + column;
      used[j] = valid_query && key < sequence_length;
      if (used[j]) {
        used[j] = mask.admits(key, query, query_position);
      }
      scores[j] = used[j] ? float(dots[j]) * query_scale * attention_gemm_scales[column] : NEGATIVE_INFINITY;
      block_max = max(block_max, scores[j]);
    }
    block_max = segment_reduce<Max<float>, GEMM_THREADS>(block_max, GEMM_ROW_LANES);
    let new_max = max(row_max, block_max);
    let factor = new_max == NEGATIVE_INFINITY ? 0.0 : exp2(row_max - new_max);
    row_max = new_max;
    var block_sum = 0.0;
    [unroll]
    for (uint j = 0; j < ATTENTION_GEMM_BLOCK_KEYS / GEMM_ROW_LANES; j++) {
      // Rejected keys weigh nothing, so a row without an admitted key sums to 0.
      let weight = used[j] ? exp2(scores[j] - new_max) : 0.0;
      attention_gemm_weights[row * ATTENTION_GEMM_BLOCK_KEYS + lane + GEMM_ROW_LANES * j] = weight;
      block_sum += weight;
    }
    row_sum = row_sum * factor + segment_reduce<Sum<float>, GEMM_THREADS>(block_sum, GEMM_ROW_LANES);
    [unroll]
    for (uint i = 0; i < GEMM_MAX_HEAD_DIM / GEMM_ROW_LANES; i++) {
      output[i] *= factor;
    }

    for (uint first_value = 0; first_value < ATTENTION_GEMM_BLOCK_KEYS; first_value += GEMM_VALUE_KEYS) {
      GroupMemoryBarrierWithGroupSync();
      attention_gemm_stage(
        v_codes, v_scales, kv_head, num_kv_heads, first_key + first_value, GEMM_VALUE_KEYS, sequence_length, HEAD_DIM,
        invocation, true
      );
      GroupMemoryBarrierWithGroupSync();

      for (uint key = 0; key < GEMM_VALUE_KEYS; key++) {
        let weight =
          attention_gemm_weights[row * ATTENTION_GEMM_BLOCK_KEYS + first_value + key] * attention_gemm_scales[key];
        [unroll]
        for (uint m = 0; m < GEMM_MAX_HEAD_DIM / (4 * GEMM_ROW_LANES); m++) {
          if (m < HEAD_DIM / (4 * GEMM_ROW_LANES)) {
            let values = asfloat(attention_gemm_block[key * (HEAD_DIM / 4) + lane + GEMM_ROW_LANES * m]);
            [unroll]
            for (uint e = 0; e < 4; e++) {
              output[4 * m + e] += weight * values[e];
            }
          }
        }
      }
    }
  }

  if (valid_query) {
    let partial = (query_rows + group_row) * blocks + split;
    if (lane == 0) {
      sums[partial] = row_sum;
      // The maximum in the merge's base.
      maxs[partial] = row_max / GEMM_LOG2E;
    }
    [unroll]
    for (uint m = 0; m < GEMM_MAX_HEAD_DIM / (4 * GEMM_ROW_LANES); m++) {
      if (m < HEAD_DIM / (4 * GEMM_ROW_LANES)) {
        [unroll]
        for (uint e = 0; e < 4; e++) {
          out[partial * HEAD_DIM + 4 * (lane + GEMM_ROW_LANES * m) + e] = output[4 * m + e];
        }
      }
    }
  }
}

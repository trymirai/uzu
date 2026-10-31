import definitions;
import attention.mask;
import common.workgroup_reduction;
import generated.attention_gemm;
import generated.ring;
import generated.trie;

// Metal's AttentionGemm on scalar tiles: a workgroup takes ATTENTION_GEMM_QUERIES queries of one head, staged scaled by
// scale * log2(e), and walks their keys in blocks of BLOCK_KEYS, each staged in turn as keys then values. The run of
// GEMM_ROW_LANES invocations at (subgroup order) 4 r holds query row r: key column `lane` + 4 j of a block's scores,
// output dimensions 4 (lane + 4 m) to 4 (lane + 4 m) + 3, and the row's base-2 online softmax.

static const uint GEMM_THREADS = 128;
static const uint GEMM_ROW_LANES = 4;
static const uint GEMM_MAX_HEAD_DIM = 256;
static const uint GEMM_MAX_BLOCK_KEYS = 32;
// Rows of 16-bit elements in 32-bit words, padded by 16 bytes.
static const uint GEMM_QUERY_WORDS = ATTENTION_GEMM_QUERIES * (GEMM_MAX_HEAD_DIM / 2 + 4);
static const uint GEMM_BLOCK_WORDS = 2176;
static const uint GEMM_WEIGHTS = ATTENTION_GEMM_QUERIES * GEMM_MAX_BLOCK_KEYS;
static const float GEMM_LOG2E = 1.4426950408889634;
static const float GEMM_MASKED_SCORE = -1e9 * GEMM_LOG2E;

groupshared uint attention_gemm_queries[GEMM_QUERY_WORDS];
groupshared uint attention_gemm_block[GEMM_BLOCK_WORDS];
groupshared float attention_gemm_weights[GEMM_WEIGHTS];

// The keys a block stages: 16 for 256 dimensions, so the tiles fit 32 KB, else 32.
uint attention_gemm_block_keys(uint head_dim) {
  return head_dim > 128 ? 16 : 32;
}

float4 attention_gemm_unpack<T : IModelFloat16>(uint2 words) {
  return float4(
    T.fromStorageBits(words.x & 0xFFFF).toFloat(),
    T.fromStorageBits(words.x >> 16).toFloat(),
    T.fromStorageBits(words.y & 0xFFFF).toFloat(),
    T.fromStorageBits(words.y >> 16).toFloat()
  );
}

uint2 attention_gemm_shared(uint first_word, bool queries) {
  if (queries) {
    return uint2(attention_gemm_queries[first_word], attention_gemm_queries[first_word + 1]);
  }
  return uint2(attention_gemm_block[first_word], attention_gemm_block[first_word + 1]);
}

// Block `block` of `rows` rows of `source` at `stride` into the block tile, rows at or past `valid_rows` zero.
void attention_gemm_stage<T : IModelFloat16>(
  Ptr<T, Access.Read> source,
  uint base,
  uint stride,
  uint first_row,
  uint rows,
  uint valid_rows,
  uint head_dim,
  uint invocation
) {
  let chunks = head_dim / 4;
  let row_words = head_dim / 2 + 4;
  for (uint chunk = invocation; chunk < rows * chunks; chunk += GEMM_THREADS) {
    let row = chunk / chunks;
    let column = chunk % chunks;
    var words = uint2(0, 0);
    if (first_row + row < valid_rows) {
      words = load_aligned<uint2, 8>(source, base + (first_row + row) * stride + 4 * column);
    }
    attention_gemm_block[row * row_words + 2 * column] = words.x;
    attention_gemm_block[row * row_words + 2 * column + 1] = words.y;
  }
}

[[Kernel]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256")]]
[[Precondition("new", "context.physical_device().properties.limits.max_compute_shared_memory_size >= crate::backends::vulkan::gpu_types::attention_gemm::ATTENTION_GEMM_SHARED_BYTES")]]
[[Precondition("encode", "gqa_factor > 0")]]
[[Precondition("encode", "suffix_length <= sequence_length")]]
func AttentionGemm<T: IModelFloat16>(
  Ptr<T, Access.Read> queries,
  Ptr<T, Access.Read> keys,
  Ptr<T, Access.Read> values,
  Ptr<T, Access.ReadWrite> out,
  uint gqa_factor,
  uint sequence_length,
  uint k_head_stride,
  uint k_seq_stride,
  uint v_head_stride,
  uint v_seq_stride,
  [[Optional("is_kv_cache_ring")]] RingParams ring_params,
  float scale,
  [[Optional("is_trie")]] Ptr<TrieNode, Access.Read> trie,
  [[Optional("is_sliding_window")]] uint sliding_window_size,
  [[Optional("has_sinks")]] Ptr<T, Access.Read> sinks,
  uint num_heads,
  uint suffix_length,
  [[Groups("suffix_length.div_ceil(crate::backends::vulkan::gpu_types::attention_gemm::ATTENTION_GEMM_QUERIES)")]] uint tile,
  [[Groups("num_heads")]] uint head,
  [[Threads("128")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool has_sinks,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
) {
  static_assert(GEMM_THREADS == ATTENTION_GEMM_QUERIES * GEMM_ROW_LANES, "four invocations a query row");
  static_assert(
    GEMM_BLOCK_WORDS >= 16 * (256 / 2 + 4) && GEMM_BLOCK_WORDS >= 32 * (128 / 2 + 4),
    "a block of every head dimension's keys"
  );
  static_assert(
    4 * (GEMM_QUERY_WORDS + GEMM_BLOCK_WORDS + GEMM_WEIGHTS) == ATTENTION_GEMM_SHARED_BYTES,
    "the shared memory the host checks"
  );
  let invocation = subgroup_invocation();
  let row = invocation / GEMM_ROW_LANES;
  let lane = invocation % GEMM_ROW_LANES;
  let block_keys = attention_gemm_block_keys(HEAD_DIM);
  let row_words = HEAD_DIM / 2 + 4;
  let first_query = tile * ATTENTION_GEMM_QUERIES;
  let query = first_query + row;
  let mask = AttentionMask(
    sequence_length, suffix_length, ring_params, trie, sliding_window_size, is_kv_cache_ring, is_causal, is_trie,
    is_sliding_window
  );
  let query_position = mask.query_position(min(query, suffix_length - 1));
  let kv_head = head / gqa_factor;

  // The queries, scaled in T as Metal scales them.
  let query_scale = T.fromFloat(scale * GEMM_LOG2E).toFloat();
  let chunks = HEAD_DIM / 4;
  for (uint chunk = invocation; chunk < ATTENTION_GEMM_QUERIES * chunks; chunk += GEMM_THREADS) {
    let chunk_row = chunk / chunks;
    let column = chunk % chunks;
    var words = uint2(0, 0);
    if (first_query + chunk_row < suffix_length) {
      let source = load_aligned<uint2, 8>(queries, ((head * suffix_length) + first_query + chunk_row) * HEAD_DIM + 4 * column);
      let scaled = attention_gemm_unpack<T>(source) * query_scale;
      words.x = T.fromFloat(scaled.x).storageBits() | (T.fromFloat(scaled.y).storageBits() << 16);
      words.y = T.fromFloat(scaled.z).storageBits() | (T.fromFloat(scaled.w).storageBits() << 16);
    }
    attention_gemm_queries[chunk_row * row_words + 2 * column] = words.x;
    attention_gemm_queries[chunk_row * row_words + 2 * column + 1] = words.y;
  }

  var row_max = NEGATIVE_INFINITY;
  var row_sum = 0.0;
  if (has_sinks) {
    row_max = sinks[head].toFloat() * GEMM_LOG2E;
    row_sum = 1.0;
  }
  float output[GEMM_MAX_HEAD_DIM / GEMM_ROW_LANES];
  [unroll]
  for (uint i = 0; i < GEMM_MAX_HEAD_DIM / GEMM_ROW_LANES; i++) {
    output[i] = 0.0;
  }

  // Under causality no query of the tile sees a key past its last query's position.
  var blocks = (sequence_length + block_keys - 1) / block_keys;
  if (is_causal) {
    let last_key = mask.prefix_length + min(first_query + ATTENTION_GEMM_QUERIES, suffix_length) - 1;
    blocks = min(blocks, last_key / block_keys + 1);
  }
  for (uint block = 0; block < blocks; block++) {
    let first_key = block * block_keys;
    GroupMemoryBarrierWithGroupSync();
    attention_gemm_stage<T>(keys, kv_head * k_head_stride, k_seq_stride, first_key, block_keys, sequence_length, HEAD_DIM, invocation);
    GroupMemoryBarrierWithGroupSync();

    float scores[GEMM_MAX_BLOCK_KEYS / GEMM_ROW_LANES];
    [unroll]
    for (uint j = 0; j < GEMM_MAX_BLOCK_KEYS / GEMM_ROW_LANES; j++) {
      scores[j] = 0.0;
    }
    for (uint word = 0; word < HEAD_DIM / 2; word += 2) {
      let q = attention_gemm_unpack<T>(attention_gemm_shared(row * row_words + word, true));
      [unroll]
      for (uint j = 0; j < GEMM_MAX_BLOCK_KEYS / GEMM_ROW_LANES; j++) {
        if (j < block_keys / GEMM_ROW_LANES) {
          let k = attention_gemm_unpack<T>(attention_gemm_shared((lane + GEMM_ROW_LANES * j) * row_words + word, false));
          scores[j] += dot(q, k);
        }
      }
    }

    var block_max = GEMM_MASKED_SCORE;
    [unroll]
    for (uint j = 0; j < GEMM_MAX_BLOCK_KEYS / GEMM_ROW_LANES; j++) {
      if (j < block_keys / GEMM_ROW_LANES) {
        let key = first_key + lane + GEMM_ROW_LANES * j;
        var used = query < suffix_length && key < sequence_length;
        if (used) {
          used = mask.admits(key, query, query_position);
        }
        scores[j] = used ? scores[j] : GEMM_MASKED_SCORE;
        block_max = max(block_max, scores[j]);
      }
    }
    block_max = segment_reduce<Max<float>, GEMM_THREADS>(block_max, GEMM_ROW_LANES);
    let new_max = max(row_max, block_max);
    let factor = exp2(row_max - new_max);
    row_max = new_max;
    var block_sum = 0.0;
    [unroll]
    for (uint j = 0; j < GEMM_MAX_BLOCK_KEYS / GEMM_ROW_LANES; j++) {
      if (j < block_keys / GEMM_ROW_LANES) {
        let weight = exp2(scores[j] - new_max);
        attention_gemm_weights[row * block_keys + lane + GEMM_ROW_LANES * j] = weight;
        block_sum += weight;
      }
    }
    row_sum = row_sum * factor + segment_reduce<Sum<float>, GEMM_THREADS>(block_sum, GEMM_ROW_LANES);
    [unroll]
    for (uint i = 0; i < GEMM_MAX_HEAD_DIM / GEMM_ROW_LANES; i++) {
      output[i] *= factor;
    }

    GroupMemoryBarrierWithGroupSync();
    attention_gemm_stage<T>(values, kv_head * v_head_stride, v_seq_stride, first_key, block_keys, sequence_length, HEAD_DIM, invocation);
    GroupMemoryBarrierWithGroupSync();

    for (uint key = 0; key < block_keys; key++) {
      let weight = attention_gemm_weights[row * block_keys + key];
      [unroll]
      for (uint m = 0; m < GEMM_MAX_HEAD_DIM / 16; m++) {
        if (m < HEAD_DIM / 16) {
          let v = attention_gemm_unpack<T>(attention_gemm_shared(key * row_words + 2 * (lane + GEMM_ROW_LANES * m), false));
          output[4 * m] += weight * v.x;
          output[4 * m + 1] += weight * v.y;
          output[4 * m + 2] += weight * v.z;
          output[4 * m + 3] += weight * v.w;
        }
      }
    }
  }

  if (query < suffix_length) {
    let inverse_sum = 1.0 / row_sum;
    let first = (query * num_heads + head) * HEAD_DIM;
    [unroll]
    for (uint m = 0; m < GEMM_MAX_HEAD_DIM / 16; m++) {
      if (m < HEAD_DIM / 16) {
        [unroll]
        for (uint e = 0; e < 4; e++) {
          out[first + 4 * (lane + GEMM_ROW_LANES * m) + e] = T.fromFloat(output[4 * m + e] * inverse_sum);
        }
      }
    }
  }
}

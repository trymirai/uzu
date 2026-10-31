import definitions;
import attention.attention;
import attention.mask;
import generated.kv_layout;
import generated.ring;
import generated.trie;

// One query of one head over the INT8 keys and values of its key-value head, the keys its mask admits.
struct CodeKeys : IAttentionKeys {
  Ptr<int8_t, Access.Read> q_codes;
  Ptr<bf16, Access.Read> q_scales;
  Ptr<int8_t, Access.Read> k_codes;
  Ptr<bf16, Access.Read> k_scales;
  Ptr<uint8_t, Access.Read> v_codes;
  Ptr<bf16, Access.Read> v_scales;
  uint head_dim;
  uint query_row;
  uint num_kv_heads;
  uint kv_head;
  AttentionMask mask;
  uint query_index;
  uint query_position;

  float query(uint j) {
    return float(q_codes[query_row * head_dim + j]);
  }

  float query_scale() {
    return q_scales[query_row].toFloat();
  }

  bool used(uint key) {
    return mask.admits(key, query_index, query_position);
  }

  float key(uint key, uint j) {
    return float(k_codes[(key * num_kv_heads + kv_head) * head_dim + j]);
  }

  float key_scale(uint key) {
    return k_scales[key * num_kv_heads + kv_head].toFloat();
  }

  float value(uint key, uint j) {
    return float(int(v_codes[(key * num_kv_heads + kv_head) * head_dim + j]) - int(V_CODE_BIAS));
  }

  float value_scale(uint key) {
    return v_scales[key * num_kv_heads + kv_head].toFloat();
  }
}

// One workgroup per query and head, writing AttentionTwoPass1's partial of a single block; keys the mask rejects are
// never read.
[[Kernel]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512")]]
[[Precondition("encode", "gqa_factor > 0")]]
[[Precondition("encode", "suffix_length <= sequence_length")]]
func AttentionSinglePass(
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
  [[Groups("num_heads")]] uint head,
  [[Groups("suffix_length")]] uint query,
  [[Threads("1024")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
) {
  static_assert(ATTENTION_THREADS == 1024, "one workgroup of Threads(1024) per head and query");
  CodeKeys source;
  source.q_codes = q_codes;
  source.q_scales = q_scales;
  source.k_codes = k_codes;
  source.k_scales = k_scales;
  source.v_codes = v_codes;
  source.v_scales = v_scales;
  source.head_dim = HEAD_DIM;
  source.query_row = head * suffix_length + query;
  source.num_kv_heads = num_kv_heads;
  source.kv_head = head / gqa_factor;
  source.mask = AttentionMask(
    sequence_length, suffix_length, ring_params, trie, sliding_window_size, is_kv_cache_ring, is_causal, is_trie,
    is_sliding_window
  );
  source.query_index = query;
  source.query_position = source.mask.query_position(query);
  let statistics = attention_single_query(source, HEAD_DIM, sequence_length, scale);
  let partial = source.query_row;
  let invocation = subgroup_invocation();
  if (invocation < HEAD_DIM) {
    out[partial * HEAD_DIM + invocation] = attention_numerator(invocation);
  }
  if (invocation == 0) {
    maxs[partial] = statistics.x;
    sums[partial] = statistics.y;
  }
}

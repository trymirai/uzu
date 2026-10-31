import definitions;
import attention.attention;
import attention.mask;
import generated.ring;
import generated.trie;

// One query of one head over strided keys and values, the keys its mask admits.
struct StridedKeys<T: IModelFloat> : IAttentionKeys {
  Ptr<T, Access.Read> queries;
  Ptr<T, Access.Read> keys;
  Ptr<T, Access.Read> values;
  uint query_base;
  uint key_base;
  uint k_seq_stride;
  uint value_base;
  uint v_seq_stride;
  AttentionMask mask;
  uint query_index;
  uint query_position;

  float query(uint j) {
    return queries[query_base + j].toFloat();
  }

  bool used(uint key) {
    return mask.admits(key, query_index, query_position);
  }

  float key(uint key, uint j) {
    return keys[key_base + key * k_seq_stride + j].toFloat();
  }

  float value(uint key, uint j) {
    return values[value_base + key * v_seq_stride + j].toFloat();
  }
}

// One workgroup per query and head; keys the mask rejects are never read.
[[Variants("T", "bf16")]]
[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512")]]
[[Precondition("encode", "gqa_factor > 0")]]
[[Precondition("encode", "suffix_length <= sequence_length")]]
func AttentionSinglePass<T: IModelFloat>(
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
  [[Groups("num_heads")]] uint head,
  [[Groups("suffix_length")]] uint query,
  [[Threads("1024")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool has_sinks,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
) {
  static_assert(ATTENTION_THREADS == 1024, "one workgroup of Threads(1024) per head and query");
  StridedKeys<T> source;
  source.queries = queries;
  source.keys = keys;
  source.values = values;
  source.query_base = (head * suffix_length + query) * HEAD_DIM;
  source.key_base = head / gqa_factor * k_head_stride;
  source.k_seq_stride = k_seq_stride;
  source.value_base = head / gqa_factor * v_head_stride;
  source.v_seq_stride = v_seq_stride;
  source.mask = AttentionMask(
    sequence_length, suffix_length, ring_params, trie, sliding_window_size, is_kv_cache_ring, is_causal, is_trie,
    is_sliding_window
  );
  source.query_index = query;
  source.query_position = source.mask.query_position(query);
  var sink = 0.0;
  if (has_sinks) {
    sink = sinks[head].toFloat();
  }
  attention_single_query(
    source, HEAD_DIM, sequence_length, scale, has_sinks, sink, out, (query * num_heads + head) * HEAD_DIM
  );
}

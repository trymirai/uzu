import definitions;
import attention.attention;
import attention.mask;
import generated.ring;
import generated.trie;

// One query of one head over strided keys and values, the keys admitted by the CPU mask.
struct StridedKeys<T: IModelFloat> : IAttentionKeys {
  Ptr<T, Access.Read> queries;
  Ptr<T, Access.Read> keys;
  Ptr<T, Access.Read> values;
  uint query_base;
  uint key_base;
  uint k_seq_stride;
  uint value_base;
  uint v_seq_stride;
  uint query_index;
  uint prefix_length;
  uint suffix_position;
  uint query_position;
  RingParams ring_params;
  Ptr<TrieNode, Access.Read> trie;
  uint sliding_window_size;
  bool is_kv_cache_ring;
  bool is_causal;
  bool is_trie;
  bool is_sliding_window;

  float query(uint j) {
    return queries[query_base + j].toFloat();
  }

  bool used(uint key) {
    return attention_use_key(
      key, query_index, prefix_length, suffix_position, query_position, ring_params, trie, sliding_window_size,
      is_kv_cache_ring, is_causal, is_trie, is_sliding_window
    );
  }

  float key(uint key, uint j) {
    return keys[key_base + key * k_seq_stride + j].toFloat();
  }

  float value(uint key, uint j) {
    return values[value_base + key * v_seq_stride + j].toFloat();
  }
}

// One workgroup per query and head; keys the CPU mask rejects are never read.
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
  let prefix_length = sequence_length - suffix_length;
  let suffix_position = is_kv_cache_ring ? ring_params.ring_length : prefix_length;
  var query_position = suffix_position + query;
  if (is_trie) {
    query_position = suffix_position + trie[query].height;
  }
  StridedKeys<T> source;
  source.queries = queries;
  source.keys = keys;
  source.values = values;
  source.query_base = (head * suffix_length + query) * HEAD_DIM;
  source.key_base = head / gqa_factor * k_head_stride;
  source.k_seq_stride = k_seq_stride;
  source.value_base = head / gqa_factor * v_head_stride;
  source.v_seq_stride = v_seq_stride;
  source.query_index = query;
  source.prefix_length = prefix_length;
  source.suffix_position = suffix_position;
  source.query_position = query_position;
  source.ring_params = ring_params;
  source.trie = trie;
  source.sliding_window_size = sliding_window_size;
  source.is_kv_cache_ring = is_kv_cache_ring;
  source.is_causal = is_causal;
  source.is_trie = is_trie;
  source.is_sliding_window = is_sliding_window;
  var sink = 0.0;
  if (has_sinks) {
    sink = sinks[head].toFloat();
  }
  attention_single_query(
    source, HEAD_DIM, sequence_length, scale, has_sinks, sink, out, (query * num_heads + head) * HEAD_DIM
  );
}

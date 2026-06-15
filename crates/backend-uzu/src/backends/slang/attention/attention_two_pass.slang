import definitions;
import generated.ring;
import generated.trie;

[[Variants("T", "float, half")]]
[[Variants("HEAD_DIM", "64, 128, 256")]]
[[Public]] [[Kernel]] func AttentionTwoPass1<T: __BuiltinFloatingPointType, let HEAD_DIM: uint>(
  StructuredBuffer<T> queries,
  StructuredBuffer<T> keys,
  StructuredBuffer<T> values,
  RWStructuredBuffer<float> out,
  RWStructuredBuffer<float> sums,
  RWStructuredBuffer<float> maxs,
  uint gqa_factor,
  uint sequence_length,
  uint k_head_stride,
  uint k_seq_stride,
  uint v_head_stride,
  uint v_seq_stride,
  [[Optional("is_kv_cache_ring")]] RingParams ring_params,
  float scale,
  uint num_heads,
  uint suffix_length,
  [[Optional("is_trie")]] StructuredBuffer<TrieNode> trie,
  [[Optional("is_sliding_window")]] uint sliding_window_size,
  [[Optional("has_sinks")]] StructuredBuffer<float> sinks,
  [[Specialize]] bool has_sinks,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
  ThreadContext thread_context,
  [[Groups("num_heads.div_ceil(8)")]] uint head_group_idx,
  [[Groups("suffix_length")]] uint q_seq_idx,
  [[Groups("32")]] uint block_idx,
  [[Threads("8 * 32")]] uint tid,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Variants("HEAD_DIM", "64, 128, 256")]]
[[Public]] [[Kernel]] func AttentionTwoPass2<T: __BuiltinFloatingPointType, let HEAD_DIM: uint>(
  StructuredBuffer<float> partials,
  StructuredBuffer<float> sums,
  StructuredBuffer<float> maxs,
  RWStructuredBuffer<T> out,
  uint num_heads,
  uint suffix_length,
  ThreadContext thread_context,
  [[Groups("num_heads")]] uint head_idx,
  [[Groups("suffix_length")]] uint q_seq_idx,
  [[Threads("1024")]] uint tid,
) {
  // TODO
}
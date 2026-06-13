import definitions;
import generated.ring;
import generated.trie;

static const uint SEQUENCE_BLOCK_SIZE = 32;
static const uint HEAD_BLOCK_SIZE = 32;

[[Variants("T", "float, half")]]
[[Variants("HEAD_DIM", "64, 128, 256")]]
[[GroupShared("shared_max_scores", "SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE")]]
[[GroupShared("shared_sum_exp_scores", "SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE")]]
[[GroupShared("shared_outputs", "SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE")]]
[[Public]] [[Kernel]] func AttentionSinglePass<T: __BuiltinFloatingPointType, let HEAD_DIM: uint, shared_max_scores: IGroupShared<float>, shared_sum_exp_scores: IGroupShared<float>, shared_outputs: IGroupShared<float>>(
  StructuredBuffer<T> queries,
  StructuredBuffer<T> keys,
  StructuredBuffer<T> values,
  RWStructuredBuffer<T> out,
  uint gqa_factor,
  uint sequence_length,
  uint k_head_stride,
  uint k_seq_stride,
  uint v_head_stride,
  uint v_seq_stride,
  [[Optional("is_kv_cache_ring")]] RingParams ring_params,
  float scale,
  [[Optional("is_trie")]] StructuredBuffer<TrieNode> trie,
  [[Optional("is_sliding_window")]] uint sliding_window_size,
  [[Optional("has_sinks")]] StructuredBuffer<float> sinks,
  uint num_heads,
  uint suffix_length,
  [[Specialize]] bool has_sinks,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
  ThreadContext thread_context,
  [[Groups("num_heads")]] uint head_idx,
  [[Groups("suffix_length")]] uint q_seq_idx,
  [[Threads("1024")]] uint tid,
) {
  // TODO
}
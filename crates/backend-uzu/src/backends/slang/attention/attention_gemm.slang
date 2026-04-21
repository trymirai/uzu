import definitions;
import generated.attention;
import generated.ring;
import generated.trie;

static const uint BLOCK_QUERY_ROWS = 32;

[[Variants("T", "float, half")]]
[[Variants("BK", "16, 32")]]
[[Variants("BD", "64, 128, 256")]]
[[GroupShared("q_smem", "BLOCK_QUERY_ROWS * (BD + 16 / sizeof(T))")]]
[[GroupShared("kv_smem", "BK * (BD + 16 / sizeof(T))")]]
[[Public]] [[Kernel]] func AttentionGemm<T: __BuiltinFloatingPointType, let BK: uint, let BD: uint, q_smem: IGroupShared<T>, kv_smem: IGroupShared<T>>(
  StructuredBuffer<T> q,
  StructuredBuffer<T> k,
  StructuredBuffer<T> v,
  RWStructuredBuffer<T> o,
  AttnParams params,
  [[Optional("is_kv_cache_ring")]] RingParams ring_params,
  [[Optional("is_trie")]] StructuredBuffer<TrieNode> trie,
  [[Optional("is_sliding_window")]] uint sliding_window_size,
  [[Optional("has_sinks")]] StructuredBuffer<float> sinks,
  uint num_heads,
  uint suffix_length,
  [[Specialize]] bool align_q,
  [[Specialize]] bool align_k,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
  [[Specialize]] bool has_sinks,
  ThreadContext thread_context,
  [[Groups("suffix_length.div_ceil(32)")]] uint q_tile_idx,
  [[Groups("num_heads")]] uint head_idx,
  [[Groups("1")]] uint batch_idx,
  [[Threads("128")]] uint lid,
) {
  // TODO
}
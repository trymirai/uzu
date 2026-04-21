import definitions;
import generated.kv_cache_update;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func KVCacheUpdate<T: __BuiltinFloatingPointType>(
  RWStructuredBuffer<T> in_place_keys,
  RWStructuredBuffer<T> in_place_values,
  [[Constant]] StructuredBuffer<Swap> swaps,
  uint swap_count,
  uint num_heads,
  uint max_sequence_length,
  uint head_dim,
  [[Axis("num_heads", "32")]] uint head_idx,
  [[Axis("head_dim", "32")]] uint channel_idx,
) {
  // TODO
}
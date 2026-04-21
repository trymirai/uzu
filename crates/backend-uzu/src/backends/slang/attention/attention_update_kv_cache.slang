import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func AttentionUpdateKVCache<T: __BuiltinFloatingPointType>(
  [[Optional("!keys_in_place")]] StructuredBuffer<T> rotated_keys,
  StructuredBuffer<T> qkv,
  RWStructuredBuffer<T> key_cache,
  RWStructuredBuffer<T> value_cache,
  uint num_groups,
  uint num_heads,
  uint head_dim,
  uint suffix_length,
  uint prefix_segment_length,
  uint max_sequence_length,
  [[Axis("num_groups", "1")]] uint group_index,
  [[Axis("suffix_length", "1")]] uint token_index,
  [[Axis("head_dim", "64")]] uint dim_index,
  [[Specialize]] bool keys_in_place,
) {
  // TODO
}
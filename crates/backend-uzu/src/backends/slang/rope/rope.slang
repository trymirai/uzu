import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func Rope<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> qkv,
  StructuredBuffer<T> cosines,
  StructuredBuffer<T> sines,
  StructuredBuffer<int> token_positions,
  RWStructuredBuffer<T> rotated_queries,
  RWStructuredBuffer<T> rotated_keys,
  uint head_dim,
  uint rope_dim,
  uint num_heads,
  uint num_groups,
  uint suffix_length,
  uint max_sequence_length,
  [[Axis("num_heads", "1")]] uint head_index,
  [[Axis("suffix_length", "1")]] uint token_index,
  [[Axis("head_dim", "128")]] uint dimension_index,
) {
  // TODO
}
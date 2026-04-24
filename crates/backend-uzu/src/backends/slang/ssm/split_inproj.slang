import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func SplitInProj<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> conv_out,
  RWStructuredBuffer<T> z_out,
  RWStructuredBuffer<T> dt_out,
  StructuredBuffer<T> z_bias,
  uint suffix_length,
  uint total_dim,
  uint conv_dim,
  uint inner_dim,
  uint num_heads,
  [[Axis("suffix_length", "16")]] uint row,
  [[Axis("total_dim", "16")]] uint col,
) {
  // TODO
}
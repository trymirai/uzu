import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func PoolingCls<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint seq_len,
  uint hidden_dim,
  uint batch_size,
  [[Axis("hidden_dim", "16")]] uint dim_idx,
  [[Axis("batch_size", "16")]] uint batch_idx,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func PoolingMean<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint seq_len,
  uint hidden_dim,
  uint batch_size,
  [[Axis("hidden_dim", "16")]] uint dim_idx,
  [[Axis("batch_size", "16")]] uint batch_idx,
) {
  // TODO
}
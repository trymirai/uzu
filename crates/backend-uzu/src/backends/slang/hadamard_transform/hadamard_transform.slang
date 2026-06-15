import definitions;

[[Variants("T", "half, float")]]
[[Public]] [[Kernel]] func HadamardTransform<T: __BuiltinFloatingPointType>(
  RWStructuredBuffer<T> data,
  StructuredBuffer<int> factors,
  uint hidden_dim,
  uint batch_size,
  [[Groups("hidden_dim.div_ceil(32)")]] uint block_index,
  [[Groups("batch_size")]] uint batch_index,
  [[Threads("32")]] uint lane_index,
) {
  // TODO
}
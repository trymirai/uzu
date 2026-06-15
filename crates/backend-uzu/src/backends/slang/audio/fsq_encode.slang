import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func AudioFsqEncode<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  RWStructuredBuffer<int> tokens,
  StructuredBuffer<int> lengths,
  int num_groups,
  int seq_len,
  int codebook_dim,
  [[Constant]] StructuredBuffer<int> num_levels,
  [[Constant]] StructuredBuffer<int> dim_base_index,
  float eps,
  int batch_size,
  [[Axis("seq_len", "32")]] uint t,
  [[Axis("num_groups", "1")]] uint g,
  [[Axis("batch_size", "1")]] uint b,
) {
  // TODO
}
import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func AudioFsqDecode<T: __BuiltinFloatingPointType>(
  StructuredBuffer<int> tokens,
  RWStructuredBuffer<T> out,
  StructuredBuffer<int> lengths,
  int num_groups,
  int seq_len,
  int codebook_dim,
  [[Constant]] StructuredBuffer<int> num_levels,
  [[Constant]] StructuredBuffer<int> dim_base_index,
  int batch_size,
  [[Axis("seq_len", "32")]] uint t,
  [[Axis("num_groups * codebook_dim", "1")]] uint gd,
  [[Axis("batch_size", "1")]] uint b,
) {
  // TODO
}
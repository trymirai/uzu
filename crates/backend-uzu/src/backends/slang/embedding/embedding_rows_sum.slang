import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func EmbeddingRowsSum<T: __BuiltinFloatingPointType>(
  StructuredBuffer<uint> token_indices,
  StructuredBuffer<T> weights,
  RWStructuredBuffer<T> output,
  uint num_rows,
  uint total_rows,
  uint model_dim,
  uint codebook_stride,
  [[Axis("model_dim", "256")]] uint dim_idx,
) {
  // TODO
}
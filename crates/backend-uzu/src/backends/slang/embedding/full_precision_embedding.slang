import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func FullPrecisionEmbeddingLookup<T: __BuiltinFloatingPointType>(
  StructuredBuffer<uint64_t> token_ids,
  StructuredBuffer<T> weights,
  RWStructuredBuffer<T> output,
  uint batch_size,
  uint vocab_size,
  uint model_dim,
  float input_scale,
  [[Axis("model_dim", "256")]] uint dim_idx,
  [[Axis("batch_size", "1")]] uint batch_idx,
) {
  // TODO
}
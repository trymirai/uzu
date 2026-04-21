import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func QuantizedEmbeddingLookup<T: __BuiltinFloatingPointType>(
  StructuredBuffer<uint64_t> token_ids,
  StructuredBuffer<uint8_t> weights,
  StructuredBuffer<T> scales,
  StructuredBuffer<T> biases,
  RWStructuredBuffer<T> output,
  uint batch_size,
  uint vocab_size,
  uint model_dim,
  float input_scale,
  [[Specialize]] uint group_size,
  [[Specialize]] uint quant_mode,
  [[Axis("model_dim", "16")]] uint dim_idx,
  [[Axis("batch_size", "16")]] uint batch_idx,
) {
  // TODO
}
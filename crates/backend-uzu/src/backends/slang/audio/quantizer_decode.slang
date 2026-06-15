import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func AudioQuantizerDecode<T: __BuiltinFloatingPointType>(
  StructuredBuffer<uint> tokens,
  StructuredBuffer<int> lengths,
  StructuredBuffer<T> semantic_codebook,
  StructuredBuffer<T> semantic_out_proj,
  StructuredBuffer<T> semantic_out_bias,
  StructuredBuffer<T> residual_codebooks,
  StructuredBuffer<T> residual_out_proj,
  StructuredBuffer<T> residual_out_bias,
  RWStructuredBuffer<T> output,
  int batch_size,
  int total_codebooks,
  int seq_len,
  int input_dim,
  int codebook_dim,
  int residual_quantizers,
  int semantic_cardinality,
  int residual_cardinality,
  [[Axis("input_dim", "32")]] uint d,
  [[Axis("seq_len", "1")]] uint t,
  [[Axis("batch_size", "1")]] uint b,
) {
  // TODO
}
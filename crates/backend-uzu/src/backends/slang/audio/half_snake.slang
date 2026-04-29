import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func AudioHalfSnake<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  StructuredBuffer<T> alpha,
  RWStructuredBuffer<T> output,
  int channels,
  int seq_len,
  int snake_channels,
  float negative_slope,
  float eps,
  int batch_size,
  [[Axis("seq_len", "32")]] uint t,
  [[Axis("channels", "1")]] uint c,
  [[Axis("batch_size", "1")]] uint b,
) {
  // TODO
}
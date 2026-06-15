import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func TensorAddScale<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  StructuredBuffer<T> bias,
  RWStructuredBuffer<T> output,
  uint num_cols,
  uint length,
  float scale,
  [[Axis("length", "32")]] uint position,
) {
  // TODO
}
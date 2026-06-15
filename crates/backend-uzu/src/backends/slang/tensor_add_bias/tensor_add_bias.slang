import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func TensorAddBias<T: __BuiltinFloatingPointType>(
  [[Optional("!in_place")]] StructuredBuffer<T> input,
  StructuredBuffer<T> bias,
  RWStructuredBuffer<T> output,
  uint num_cols,
  uint length,
  [[Axis("length", "32")]] uint position,
  [[Specialize]] bool in_place,
) {
  // TODO
}
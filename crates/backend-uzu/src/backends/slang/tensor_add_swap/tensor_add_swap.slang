import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func TensorAddSwap<T: __BuiltinFloatingPointType>(
  RWStructuredBuffer<T> skip_buffer,
  RWStructuredBuffer<T> main_buffer,
  uint length,
  [[Axis("length", "256")]] uint position,
) {
  // TODO
}
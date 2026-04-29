import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func TensorCopy<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> src_buffer,
  RWStructuredBuffer<T> dst_buffer,
  uint length,
  [[Axis("length", "256")]] uint position,
) {
  // TODO
}
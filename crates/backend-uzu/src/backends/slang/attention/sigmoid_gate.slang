import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func SigmoidGate<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> gate,
  RWStructuredBuffer<T> output,
  uint total_elements,
  [[Axis("total_elements", "256")]] uint tid,
) {
  // TODO
}
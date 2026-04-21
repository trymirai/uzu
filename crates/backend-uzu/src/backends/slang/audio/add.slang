import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func AudioAdd<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> a,
  StructuredBuffer<T> b,
  RWStructuredBuffer<T> out,
  int n,
  [[Axis("n", "256")]] uint tid,
) {
  // TODO
}
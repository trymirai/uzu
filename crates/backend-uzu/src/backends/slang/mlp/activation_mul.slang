import definitions;
import generated.activation_type;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func MlpGateActMul<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> fused_up,
  RWStructuredBuffer<T> hidden,
  [[Optional("use_hadamard")]] StructuredBuffer<int> hadamard_factors,
  int h,
  int m,
  ActivationType act_type,
  [[Specialize]] bool use_hadamard,
  [[Axis("h", "64")]] uint j,
  [[Axis("m", "1")]] uint row,
) {
  // TODO
}
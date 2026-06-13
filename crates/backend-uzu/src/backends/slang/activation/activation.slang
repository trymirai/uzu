import definitions;
import generated.activation_type;

[[Variants("T", "half, float")]]
[[Public]] [[Kernel]] func Activation<T: __BuiltinFloatingPointType>(
  [[Optional("!in_place")]] StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint n,
  ActivationType act_type,
  [[Specialize]] bool in_place,
  [[Axis("n", "256")]] uint tid,
) {
  // TODO
}
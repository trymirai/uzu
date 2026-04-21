import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func DeltaNetConvUpdate<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> conv_weight,
  [[Optional("has_bias")]] StructuredBuffer<T> bias,
  RWStructuredBuffer<T> in_out,
  RWStructuredBuffer<T> state,
  uint kernel_size,
  uint conv_dim,
  uint state_stride,
  [[Specialize]] bool has_bias,
  [[Axis("conv_dim", "256")]] uint channel_idx,
) {
  // TODO
}
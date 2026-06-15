import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func DeltaNetConvScan<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> conv_padded,
  StructuredBuffer<T> conv_weight,
  [[Optional("has_bias")]] StructuredBuffer<T> bias,
  RWStructuredBuffer<T> in_proj,
  RWStructuredBuffer<T> state_out,
  uint suffix_len,
  uint kernel_size,
  uint row_stride,
  uint state_stride,
  uint conv_dim,
  uint out_stride,
  [[Specialize]] bool has_bias,
  [[Axis("suffix_len + kernel_size - 1", "1")]] uint token_idx,
  [[Axis("conv_dim", "32")]] uint channel_idx,
) {
  // TODO
}
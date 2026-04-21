import definitions;
import generated.activation_type;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func Conv1dPack<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> state_in,
  StructuredBuffer<T> x,
  RWStructuredBuffer<T> padded,
  uint state_stride,
  uint row_stride,
  uint suffix_len,
  uint num_channels,
  [[Axis("num_channels", "32")]] uint channel_idx,
  [[Axis("state_stride + suffix_len", "1")]] uint row_idx,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func Conv1dDecode<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> x,
  StructuredBuffer<T> w,
  [[Optional("has_bias")]] StructuredBuffer<T> b,
  [[Optional("!state_in_place")]] StructuredBuffer<T> state,
  RWStructuredBuffer<T> x_out,
  RWStructuredBuffer<T> b_out,
  RWStructuredBuffer<T> c_out,
  RWStructuredBuffer<T> next_state,
  uint kernel_size,
  uint row_stride,
  uint state_stride,
  uint num_channels,
  uint suffix_len,
  uint inner_dim,
  uint proj_dim,
  ActivationType activation_type,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool state_in_place,
  [[Axis("suffix_len", "32")]] uint token_idx,
  [[Axis("num_channels", "1")]] uint channel_idx,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func Conv1dScan<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> padded,
  StructuredBuffer<T> w,
  [[Optional("has_bias")]] StructuredBuffer<T> b,
  RWStructuredBuffer<T> x_out,
  RWStructuredBuffer<T> b_out,
  RWStructuredBuffer<T> c_out,
  RWStructuredBuffer<T> state_out,
  uint suffix_len,
  uint kernel_size,
  uint row_stride,
  uint state_stride,
  uint num_channels,
  uint inner_dim,
  uint proj_dim,
  ActivationType activation_type,
  [[Specialize]] bool has_bias,
  [[Axis("suffix_len + kernel_size.saturating_sub(1)", "32")]] uint token_idx,
  [[Axis("num_channels", "1")]] uint channel_idx,
) {
  // TODO
}
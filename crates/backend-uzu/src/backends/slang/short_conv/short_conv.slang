import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func ShortConvPack<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> state_in,
  StructuredBuffer<T> in_proj,
  RWStructuredBuffer<T> padded,
  uint state_stride,
  uint suffix_len,
  uint in_proj_stride,
  uint model_dim,
  [[Axis("model_dim", "256")]] uint channel_idx,
  [[Axis("state_stride + suffix_len", "1")]] uint row_idx,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func ShortConvPrefill<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> padded,
  StructuredBuffer<T> in_proj,
  StructuredBuffer<T> w,
  [[Optional("has_bias")]] StructuredBuffer<T> b,
  RWStructuredBuffer<T> out,
  RWStructuredBuffer<T> state_out,
  uint suffix_len,
  uint kernel_size,
  uint in_proj_stride,
  uint state_stride,
  uint model_dim,
  [[Specialize]] bool has_bias,
  [[Axis("suffix_len + kernel_size.saturating_sub(1)", "1")]] uint token_idx,
  [[Axis("model_dim", "256")]] uint channel_idx,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func ShortConvDecode<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> in_proj,
  StructuredBuffer<T> w,
  [[Optional("has_bias")]] StructuredBuffer<T> b,
  [[Optional("!state_in_place")]] StructuredBuffer<T> state,
  RWStructuredBuffer<T> out,
  RWStructuredBuffer<T> next_state,
  uint suffix_len,
  uint kernel_size,
  uint in_proj_stride,
  uint state_stride,
  uint model_dim,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool state_in_place,
  [[Axis("suffix_len", "1")]] uint token_idx,
  [[Axis("model_dim", "256")]] uint channel_idx,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func ShortConvTrie<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> in_proj,
  StructuredBuffer<T> w,
  [[Optional("has_bias")]] StructuredBuffer<T> b,
  StructuredBuffer<T> base_state,
  StructuredBuffer<int> parents,
  RWStructuredBuffer<T> out,
  RWStructuredBuffer<T> suffix_state,
  uint suffix_len,
  uint kernel_size,
  uint in_proj_stride,
  uint state_stride,
  uint model_dim,
  [[Specialize]] bool has_bias,
  [[Axis("model_dim", "256")]] uint channel_idx,
) {
  // TODO
}
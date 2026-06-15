import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func SSDPrefill64<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> x,
  StructuredBuffer<T> dt_raw,
  StructuredBuffer<T> b,
  StructuredBuffer<T> c,
  StructuredBuffer<T> d,
  StructuredBuffer<T> z,
  RWStructuredBuffer<T> state,
  RWStructuredBuffer<T> y,
  uint suffix_len,
  uint group_size,
  uint state_size,
  uint[3] x_strides,
  uint[2] dt_strides,
  uint[3] cb_strides,
  uint[3] state_strides,
  uint num_heads,
  uint head_dim,
  ThreadContext thread_context,
  [[Groups("num_heads * head_dim")]] uint pair_idx,
  [[Threads("32")]] uint lane_idx,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func SSDPrefill<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> x,
  StructuredBuffer<T> dt_raw,
  StructuredBuffer<T> b,
  StructuredBuffer<T> c,
  StructuredBuffer<T> d,
  StructuredBuffer<T> z,
  RWStructuredBuffer<T> state,
  RWStructuredBuffer<T> y,
  uint suffix_len,
  uint group_size,
  uint state_size,
  uint[3] x_strides,
  uint[2] dt_strides,
  uint[3] cb_strides,
  uint[3] state_strides,
  uint num_heads,
  uint head_dim,
  ThreadContext thread_context,
  [[Groups("num_heads * head_dim")]] uint pair_idx,
  [[Threads("32")]] uint lane_idx,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func SSDPrefillSequential<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> x,
  StructuredBuffer<T> dt_raw,
  StructuredBuffer<T> b,
  StructuredBuffer<T> c,
  StructuredBuffer<T> d,
  StructuredBuffer<T> z,
  RWStructuredBuffer<T> state,
  RWStructuredBuffer<T> y,
  uint suffix_len,
  uint group_size,
  uint state_size,
  uint[3] x_strides,
  uint[2] dt_strides,
  uint[3] cb_strides,
  uint[3] state_strides,
  uint channels,
  uint head_dim,
  [[Axis("channels", "32")]] uint h_idx,
  [[Axis("head_dim", "32")]] uint dh_idx,
) {
  // TODO
}
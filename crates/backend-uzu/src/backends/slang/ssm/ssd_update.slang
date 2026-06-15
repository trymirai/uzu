import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func SSDUpdate<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> x,
  StructuredBuffer<T> dt_raw,
  StructuredBuffer<T> b,
  StructuredBuffer<T> c,
  StructuredBuffer<T> d,
  StructuredBuffer<T> z,
  [[Optional("!state_in_place")]] StructuredBuffer<T> state,
  RWStructuredBuffer<T> y,
  RWStructuredBuffer<T> next_state,
  uint group_size,
  uint state_size,
  uint[3] x_strides,
  uint[2] dt_strides,
  uint[3] cb_strides,
  uint[4] state_strides,
  uint b_size,
  uint h_size,
  uint dh_size,
  [[Specialize]] bool state_in_place,
  [[Axis("b_size", "32")]] uint b_idx,
  [[Axis("h_size", "32")]] uint h_idx,
  [[Axis("dh_size", "1")]] uint dh_idx,
) {
  // TODO
}
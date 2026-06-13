import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func MoeGatherXPerm2D<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> x,
  StructuredBuffer<int> bucketed_ids,
  RWStructuredBuffer<T> x_perm,
  StructuredBuffer<uint> sumk_buf,
  uint d_model,
  uint t,
  uint k,
  [[Groups("(t * k).div_ceil(8)")]] uint tgid_x,
  [[Threads("256")]] uint lid,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func MoeGatherXPerm1D<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> x,
  StructuredBuffer<int> bucketed_ids,
  RWStructuredBuffer<T> x_perm,
  StructuredBuffer<uint> sumk_buf,
  uint d_model,
  uint t,
  uint k,
  [[Groups("(t * k).div_ceil(256)")]] uint tgid_x,
  [[Threads("256")]] uint lid,
) {
  // TODO
}
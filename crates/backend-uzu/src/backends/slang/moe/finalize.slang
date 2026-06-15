import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func MoeFinalize<T: __BuiltinFloatingPointType>(
  StructuredBuffer<int> tok2row,
  StructuredBuffer<T> probs,
  StructuredBuffer<T> y_partial,
  RWStructuredBuffer<T> y,
  uint t_count,
  uint d_model,
  uint k_input,
  [[Threads("128")]] uint lid,
  [[Groups("d_model.div_ceil(64)")]] uint tg_n,
  [[Groups("t_count.div_ceil(32)")]] uint tg_m,
) {
  // TODO
}
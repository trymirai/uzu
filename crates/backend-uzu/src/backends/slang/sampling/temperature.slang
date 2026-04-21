import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func Temperature<T: __BuiltinFloatingPointType>(
  [[Optional("!in_place")]] StructuredBuffer<T> logits,
  RWStructuredBuffer<T> processed_logits,
  uint batch_size,
  uint vocab_size,
  float temperature,
  [[Groups("vocab_size.div_ceil(1024 * 64)")]] uint group_idx,
  [[Groups("batch_size")]] uint batch_idx,
  [[Threads("1024")]] uint thread_idx,
  [[Specialize]] bool in_place,
) {
  // TODO
}
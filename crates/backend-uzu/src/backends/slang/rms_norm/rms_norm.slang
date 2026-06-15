import definitions;

static const uint METAL_SIMD_SIZE = 32;

[[Variants("InputT", "float, half")]]
[[Variants("ScaleT", "float, half")]]
[[Variants("OutputT", "float, half")]]
[[Variants("AccumT", "float, half")]]
[[GroupShared("shared_sum", "METAL_SIMD_SIZE")]]
[[Public]] [[Kernel]] func RMSNorm<InputT: __BuiltinFloatingPointType, ScaleT: __BuiltinFloatingPointType, OutputT: __BuiltinFloatingPointType, AccumT: __BuiltinFloatingPointType, shared_sum: IGroupShared<AccumT>>(
  [[Optional("!in_place")]] StructuredBuffer<InputT> input,
  StructuredBuffer<ScaleT> scales,
  RWStructuredBuffer<OutputT> output,
  [[Optional("copy_to_shortcut")]] RWStructuredBuffer<InputT> shortcut,
  [[Optional("use_hadamard")]] StructuredBuffer<int> hadamard_factors,
  uint batch_size,
  uint element_count,
  float epsilon,
  float scale_offset,
  [[Specialize]] bool in_place,
  [[Specialize]] bool full_layer,
  [[Specialize]] bool copy_to_shortcut,
  [[Specialize]] bool residual_add,
  [[Specialize]] bool use_hadamard,
  ThreadContext thread_context,
  [[Groups("batch_size")]] uint batch_idx,
  [[Threads("1024")]] uint thread_in_row,
) {
  // TODO
}
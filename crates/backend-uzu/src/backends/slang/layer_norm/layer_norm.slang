import definitions;

static const uint METAL_SIMD_SIZE = 32;

[[Variants("IN", "float, half")]]
[[Variants("SC", "float, half")]]
[[Variants("OUT", "float, half")]]
[[Variants("ACC", "float")]]
[[GroupShared("shared_mean", "METAL_SIMD_SIZE")]]
[[GroupShared("shared_variance", "METAL_SIMD_SIZE")]]
[[Public]] [[Kernel]] func LayerNorm<IN: __BuiltinFloatingPointType, SC: __BuiltinFloatingPointType, OUT: __BuiltinFloatingPointType, ACC: __BuiltinFloatingPointType, shared_mean: IGroupShared<ACC>, shared_variance: IGroupShared<ACC>>(
  [[Optional("!in_place")]] StructuredBuffer<IN> input,
  StructuredBuffer<SC> scales,
  RWStructuredBuffer<OUT> output,
  uint batch_size,
  uint model_dim,
  float epsilon,
  float scale_offset,
  uint full_layer,
  [[Specialize]] bool in_place,
  ThreadContext thread_context,
  [[Groups("batch_size")]] uint batch_idx,
  [[Threads("1024")]] uint thread_in_row,
) {
  // TODO
}
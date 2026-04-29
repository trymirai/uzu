import definitions;

[[Variants("InputT", "float, half")]]
[[Variants("ScaleT", "float, half")]]
[[Variants("OutputT", "float, half")]]
[[Variants("AccumT", "float, half")]]
[[Public]] [[Kernel]] func QKNorm<InputT: __BuiltinFloatingPointType, ScaleT: __BuiltinFloatingPointType, OutputT: __BuiltinFloatingPointType, AccumT: __BuiltinFloatingPointType>(
  [[Optional("!in_place")]] StructuredBuffer<InputT> qkv_input,
  StructuredBuffer<ScaleT> scales,
  RWStructuredBuffer<OutputT> qkv_output,
  uint batch_size,
  uint num_q_heads,
  uint num_kv_heads,
  uint head_dim,
  float epsilon,
  float scale_offset,
  uint head_offset,
  uint head_count,
  bool full_layer,
  [[Groups("batch_size")]] uint batch_idx,
  [[Groups("head_count")]] uint head_idx,
  [[Threads("32")]] uint lane_id,
  [[Specialize]] bool in_place,
) {
  // TODO
}
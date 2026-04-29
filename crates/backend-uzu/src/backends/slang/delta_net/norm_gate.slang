import definitions;

[[Variants("T", "float, half")]]
[[GroupShared("shared_scratch", "32")]]
[[Public]] [[Kernel]] func DeltaNetNormGate<T: __BuiltinFloatingPointType, shared_scratch: IGroupShared<float>>(
  RWStructuredBuffer<T> in_out,
  StructuredBuffer<T> in_proj,
  StructuredBuffer<T> norm_weight,
  uint num_v_heads,
  uint head_v_dim,
  uint value_dim,
  uint conv_dim,
  uint total_proj_dim,
  float norm_epsilon,
  uint suffix_len,
  ThreadContext thread_context,
  [[Groups("suffix_len")]] uint token_idx,
  [[Groups("num_v_heads")]] uint hv_idx,
  [[Threads("128")]] uint tid,
) {
  // TODO
}
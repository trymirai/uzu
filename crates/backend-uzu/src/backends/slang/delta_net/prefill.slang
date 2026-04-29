import definitions;

[[Variants("T", "float, half")]]
[[Variants("HEAD_K_DIM", "128")]]
[[Public]] [[Kernel]] func DeltaNetPrefill<T: __BuiltinFloatingPointType, let HEAD_K_DIM: uint>(
  StructuredBuffer<float> q_norm,
  StructuredBuffer<float> k_norm,
  StructuredBuffer<float> beta_buf,
  StructuredBuffer<float> decay_buf,
  StructuredBuffer<T> in_proj,
  RWStructuredBuffer<T> state,
  RWStructuredBuffer<T> out,
  uint num_v_heads,
  uint num_k_heads,
  uint head_v_dim,
  uint key_dim,
  uint value_dim,
  uint suffix_len,
  uint num_dv_groups,
  [[Groups("num_v_heads")]] uint hv_idx,
  [[Groups("num_dv_groups")]] uint dv_group,
  [[Threads("256")]] uint tid,
) {
  // TODO
}
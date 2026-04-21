import definitions;

[[Variants("T", "float, half")]]
[[Variants("HEAD_K_DIM", "128")]]
[[Public]] [[Kernel]] func DeltaNetPrefillPrep<T: __BuiltinFloatingPointType, let HEAD_K_DIM: uint>(
  StructuredBuffer<T> in_proj,
  StructuredBuffer<T> a_log,
  StructuredBuffer<T> dt_bias,
  RWStructuredBuffer<float> q_norm_out,
  RWStructuredBuffer<float> k_norm_out,
  RWStructuredBuffer<float> beta_out,
  RWStructuredBuffer<float> decay_out,
  uint num_v_heads,
  uint num_k_heads,
  uint key_dim,
  uint value_dim,
  uint suffix_len,
  [[Groups("suffix_len")]] uint token_idx,
  [[Groups("num_k_heads")]] uint hk_idx,
  [[Threads("32")]] uint lane,
) {
  // TODO
}
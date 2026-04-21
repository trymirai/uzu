import definitions;

static const uint METAL_SIMD_SIZE = 32;

[[Variants("T", "float, half")]]
[[Variants("HEAD_K_DIM", "128")]]
[[GroupShared("shared_qk", "512 /* UPDATE_THREADS */ * 2")]]
[[GroupShared("shared_scratch", "METAL_SIMD_SIZE")]]
[[Public]] [[Kernel]] func DeltaNetUpdate<T: __BuiltinFloatingPointType, let HEAD_K_DIM: uint, shared_qk: IGroupShared<float>, shared_scratch: IGroupShared<float>>(
  StructuredBuffer<T> in_proj,
  StructuredBuffer<T> a_log,
  StructuredBuffer<T> dt_bias,
  StructuredBuffer<T> norm_weight,
  RWStructuredBuffer<T> state,
  RWStructuredBuffer<T> out,
  uint num_v_heads,
  uint num_k_heads,
  uint head_v_dim,
  uint key_dim,
  uint value_dim,
  float norm_epsilon,
  ThreadContext thread_context,
  [[Groups("num_v_heads")]] uint hv_idx,
  [[Threads("512")]] uint tid,
) {
  // TODO
}
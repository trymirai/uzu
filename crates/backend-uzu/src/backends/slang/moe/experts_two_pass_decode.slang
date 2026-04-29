import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func MoeExpertsDecodePassA<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> x_perm,
  StructuredBuffer<uint> expert_offsets,
  StructuredBuffer<T> w13_all,
  RWStructuredBuffer<float> hidden_out,
  StructuredBuffer<T> up_biases,
  uint d_model,
  uint d_ff,
  uint e,
  float gate_clip_min,
  float gate_clip_max,
  float up_clip_min,
  float up_clip_max,
  float silu_alpha,
  StructuredBuffer<uint> tile_map,
  [[Specialize]] uint gating_sel,
  ThreadContext thread_context,
  [[Groups("INDIRECT")]] uint tile_idx,
  [[Threads("128")]] uint tid,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Variants("AccumT", "float")]]
[[Public]] [[Kernel]] func MoeExpertsDecodeDownFused2D<T: __BuiltinFloatingPointType, AccumT: __BuiltinFloatingPointType>(
  StructuredBuffer<float> hidden,
  StructuredBuffer<uint> row_expert_map,
  StructuredBuffer<T> w2_all,
  StructuredBuffer<T> down_biases,
  RWStructuredBuffer<T> y_out,
  uint total_rows,
  uint d_model,
  uint d_ff,
  uint e,
  ThreadContext thread_context,
  [[Groups("d_model.div_ceil(8)")]] uint tgpig_x,
  [[Groups("total_rows")]] uint tgpig_y,
  [[Threads("256")]] uint tid,
) {
  // TODO
}
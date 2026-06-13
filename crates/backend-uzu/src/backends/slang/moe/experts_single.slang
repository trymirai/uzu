import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func MoeExpertsDecodeSinglePassA<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> x,
  StructuredBuffer<int> topk_ids,
  StructuredBuffer<T> w13_all,
  StructuredBuffer<T> biases,
  RWStructuredBuffer<float> hidden_out,
  uint d_model,
  uint d_ff,
  uint k,
  float silu_alpha,
  float gate_clip_min,
  float gate_clip_max,
  float up_clip_min,
  float up_clip_max,
  [[Specialize]] uint gating_sel,
  ThreadContext thread_context,
  [[Groups("d_ff.div_ceil(4)")]] uint h_block_idx,
  [[Groups("k")]] uint k_slot,
  [[Threads("128")]] uint tid,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func MoeExpertsDecodeSinglePassB<T: __BuiltinFloatingPointType>(
  StructuredBuffer<float> hidden,
  StructuredBuffer<int> topk_ids,
  StructuredBuffer<T> topk_probs,
  StructuredBuffer<T> w2_all,
  StructuredBuffer<T> biases,
  RWStructuredBuffer<T> y,
  uint d_model,
  uint d_ff,
  uint k_input,
  ThreadContext thread_context,
  [[Groups("d_model.div_ceil(8)")]] uint d_block,
  [[Threads("256")]] uint tid,
) {
  // TODO
}
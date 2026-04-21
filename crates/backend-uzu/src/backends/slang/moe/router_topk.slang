import definitions;

static const uint THREADS_PER_TG = 256;
static const uint MAX_EXPERTS = 512;

[[Variants("ScalarT", "half, float")]]
[[GroupShared("x_cache", "1024")]]
[[GroupShared("logits_shared", "MAX_EXPERTS")]]
[[GroupShared("idx_shared", "MAX_EXPERTS")]]
[[GroupShared("reduce_tmp", "THREADS_PER_TG")]]
[[GroupShared("reduce_tmp_u", "THREADS_PER_TG")]]
[[GroupShared("shared_best_idx", "1")]]
[[GroupShared("shared_best_val", "1")]]
[[Public]] [[Kernel]] func MoeRouterTopK<ScalarT: __BuiltinFloatingPointType, x_cache: IGroupShared<float4>, logits_shared: IGroupShared<float>, idx_shared: IGroupShared<uint>, reduce_tmp: IGroupShared<float>, reduce_tmp_u: IGroupShared<uint>, shared_best_idx: IGroupShared<uint>, shared_best_val: IGroupShared<float>>(
  StructuredBuffer<ScalarT> input,
  StructuredBuffer<ScalarT> weight,
  StructuredBuffer<ScalarT> bias,
  RWStructuredBuffer<int> topk_ids,
  RWStructuredBuffer<ScalarT> topk_probs,
  uint t,
  uint d_model,
  uint e,
  uint k,
  bool renorm,
  ThreadContext thread_context,
  [[Groups("1")]] uint tgpig_x,
  [[Groups("t")]] uint token_idx,
  [[Threads("256")]] uint lid,
) {
  // TODO
}
import definitions;

static const uint PASSA_BM = 16;
static const uint PASSA_BN = 32;
static const uint PASSA_BK = 64;
static const uint PASSA_TG_PAD = 4;
static const uint PASSB_BM = 16;
static const uint PASSB_BN = 64;
static const uint PASSB_BK = 64;
static const uint PASSB_TG_PAD = 4;

[[Variants("T", "float, half")]]
[[GroupShared("Xs", "PASSA_BM * (PASSA_BK + PASSA_TG_PAD)")]]
[[GroupShared("Wk_up", "PASSA_BN * (PASSA_BK + PASSA_TG_PAD)")]]
[[GroupShared("Wk_gate", "PASSA_BN * (PASSA_BK + PASSA_TG_PAD)")]]
[[GroupShared("bias_up", "PASSA_BN")]]
[[GroupShared("bias_gate", "PASSA_BN")]]
[[Public]] [[Kernel]] func MoeExpertsPrefillPassA<T: __BuiltinFloatingPointType, Xs: IGroupShared<T>, Wk_up: IGroupShared<T>, Wk_gate: IGroupShared<T>, bias_up: IGroupShared<float>, bias_gate: IGroupShared<float>>(
  StructuredBuffer<T> x_perm,
  StructuredBuffer<uint> expert_offsets,
  StructuredBuffer<T> w13_all,
  StructuredBuffer<T> up_biases,
  RWStructuredBuffer<float> hidden_out,
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
  [[Groups("INDIRECT")]] uint n_tile_idx,
  [[Groups("INDIRECT")]] uint row_tile_idx,
  [[Threads("128")]] uint local_tid_x,
  ThreadContext thread_context,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[GroupShared("Hs", "PASSB_BM * (PASSB_BK + PASSB_TG_PAD)")]]
[[GroupShared("Wk", "PASSB_BN * (PASSB_BK + PASSB_TG_PAD)")]]
[[GroupShared("bias_tile", "PASSB_BN")]]
[[Public]] [[Kernel]] func MoeExpertsPrefillPassB<T: __BuiltinFloatingPointType, Hs: IGroupShared<float>, Wk: IGroupShared<T>, bias_tile: IGroupShared<float>>(
  StructuredBuffer<float> hidden,
  StructuredBuffer<uint> expert_offsets,
  StructuredBuffer<T> w2_all,
  StructuredBuffer<T> down_biases,
  RWStructuredBuffer<T> output,
  uint d_model,
  uint d_ff,
  uint e,
  StructuredBuffer<uint> tile_map,
  ThreadContext thread_context,
  [[Groups("INDIRECT")]] uint n_tile_idx,
  [[Groups("INDIRECT")]] uint row_tile_idx,
  [[Threads("128")]] uint lin,
) {
  // TODO
}
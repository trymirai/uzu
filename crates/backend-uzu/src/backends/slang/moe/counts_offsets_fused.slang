import definitions;

static const uint BLOCK_SIZE = 128;
static const uint TILE_E = 512;

[[GroupShared("tg_hist", "TILE_E")]]
[[GroupShared("scan_shared", "BLOCK_SIZE")]]
[[GroupShared("reduce_shared", "BLOCK_SIZE")]]
[[GroupShared("counts_shared", "BLOCK_SIZE")]]
[[GroupShared("carry", "1")]]
[[Public]] [[Kernel]] func MoeCountsOffsetsFused<tg_hist: IGroupShared<uint>, scan_shared: IGroupShared<uint>, reduce_shared: IGroupShared<uint>, counts_shared: IGroupShared<uint>, carry: IGroupShared<uint>>(
  StructuredBuffer<int> topk_ids,
  RWStructuredBuffer<uint> offsets,
  RWStructuredBuffer<uint> sum_k_out,
  RWStructuredBuffer<uint> partials,
  uint t_input,
  uint e_input,
  uint k_input,
  ThreadContext thread_context,
  [[Threads("128")]] uint lid,
) {
  // TODO
}
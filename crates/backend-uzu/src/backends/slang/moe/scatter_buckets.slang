import definitions;

static const uint BLOCK_SIZE = 256;
static const uint TILE_E = 512;
static const uint SIMD_WIDTH = 32;
static const uint NUM_SG = BLOCK_SIZE / SIMD_WIDTH;

[[Public]] [[Kernel]] func MoeBlockBasesFromPartials(
  StructuredBuffer<uint> partials,
  RWStructuredBuffer<uint> block_bases,
  RWStructuredBuffer<uint> block_alloc,
  uint e_input,
  uint num_blocks,
  uint num_tiles,
  uint capacity_per_expert,
  [[Axis("num_tiles * 512", "256")]] uint gid,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[GroupShared("sg_counts", "NUM_SG * TILE_E")]]
[[GroupShared("sg_base", "NUM_SG * TILE_E")]]
[[Public]] [[Kernel]] func MoeScatterBuckets<T: __BuiltinFloatingPointType, sg_counts: IGroupShared<uint>, sg_base: IGroupShared<uint>>(
  StructuredBuffer<int> topk_ids,
  StructuredBuffer<T> topk_probs,
  StructuredBuffer<uint> offsets,
  StructuredBuffer<uint> block_bases,
  StructuredBuffer<uint> block_alloc,
  RWStructuredBuffer<int> out_ids,
  RWStructuredBuffer<T> out_probs,
  uint t,
  uint e,
  uint k,
  uint num_blocks,
  uint num_tiles,
  ThreadContext thread_context,
  [[Groups("num_blocks")]] uint tgpig_x,
  [[Threads("256")]] uint lid,
) {
  // TODO
}

[[Variants("T", "float, half")]]
[[GroupShared("sg_counts", "NUM_SG * TILE_E")]]
[[GroupShared("sg_base", "NUM_SG * TILE_E")]]
[[Public]] [[Kernel]] func MoeScatterBucketsMap<T: __BuiltinFloatingPointType, sg_counts: IGroupShared<uint>, sg_base: IGroupShared<uint>>(
  StructuredBuffer<int> topk_ids,
  StructuredBuffer<T> topk_probs,
  StructuredBuffer<uint> offsets,
  StructuredBuffer<uint> block_bases,
  StructuredBuffer<uint> block_alloc,
  RWStructuredBuffer<int> out_ids,
  RWStructuredBuffer<T> out_probs,
  uint t,
  uint e,
  uint k,
  uint num_blocks,
  uint num_tiles,
  RWStructuredBuffer<int> tok2row,
  ThreadContext thread_context,
  [[Groups("num_blocks")]] uint tgpig_x,
  [[Threads("256")]] uint lid,
) {
  // TODO
}
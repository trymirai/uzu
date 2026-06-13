import definitions;

[[Public]] [[Kernel]] func MoePassATileCounts(
  StructuredBuffer<uint> expert_offsets,
  RWStructuredBuffer<uint> tile_counts,
  uint e,
  uint h_blocks,
  [[Axis("e", "256")]] uint tid,
) {
  // TODO
}

[[GroupShared("scratch", "1024")]]
[[Public]] [[Kernel]] func MoePassATileScan<scratch: IGroupShared<uint>>(
  StructuredBuffer<uint> tile_counts,
  RWStructuredBuffer<uint> tile_offsets,
  RWStructuredBuffer<uint> total_tiles,
  uint e,
  [[Groups("1")]] uint gid,
  [[Threads("1024")]] uint lid,
) {
  // TODO
}

[[Public]] [[Kernel]] func MoePassABuildRowMap(
  StructuredBuffer<uint> expert_offsets,
  RWStructuredBuffer<uint> row_expert_map,
  uint total_rows,
  uint e,
  [[Axis("total_rows", "256")]] uint tid,
) {
  // TODO
}

[[Public]] [[Kernel]] func MoePassABuildTileMap(
  StructuredBuffer<uint> expert_offsets,
  StructuredBuffer<uint> tile_offsets,
  StructuredBuffer<uint> row_expert_map,
  RWStructuredBuffer<uint> tile_map,
  uint total_rows,
  uint h_blocks,
  [[Axis("total_rows.saturating_mul(h_blocks)", "256")]] uint tid,
) {
  // TODO
}

[[Public]] [[Kernel]] func MoePassAWriteDispatchArgs(
  StructuredBuffer<uint> total_tiles,
  RWStructuredBuffer<uint> dispatch_args,
  uint num_tiles_y,
  [[Axis("1", "1")]] uint tid,
) {
  // TODO
}
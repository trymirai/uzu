import definitions;

[[Public]] [[Kernel]] func MoeTileCounts(
  StructuredBuffer<uint> offsets,
  RWStructuredBuffer<uint> tile_counts,
  uint e,
  [[Axis("e", "256")]] uint gid,
) {
  // TODO
}

[[GroupShared("scratch", "1024")]]
[[Public]] [[Kernel]] func MoeTileScan<scratch: IGroupShared<uint>>(
  StructuredBuffer<uint> tile_counts,
  RWStructuredBuffer<uint> tile_row_offsets,
  RWStructuredBuffer<uint> total_tiles_buf,
  uint e,
  [[Threads("256")]] uint tid,
) {
  // TODO
}

[[Public]] [[Kernel]] func MoeBuildTileMap(
  StructuredBuffer<uint> offsets,
  StructuredBuffer<uint> tile_row_offsets,
  StructuredBuffer<uint> tile_counts,
  RWStructuredBuffer<uint> tile_map,
  uint e,
  [[Axis("e", "256")]] uint gid,
) {
  // TODO
}

[[Public]] [[Kernel]] func MoeWriteDispatchArgs(
  StructuredBuffer<uint> total_tiles_buf,
  RWStructuredBuffer<uint> dispatch_args,
  uint num_tiles_n,
  [[Axis("1", "1")]] uint gid,
) {
  // TODO
}
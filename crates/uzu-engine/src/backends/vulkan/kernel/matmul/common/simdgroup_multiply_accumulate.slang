import definitions;

// A simdgroup lane.
public struct ThreadContext {
  // simdgroup_index * SIMD_SIZE + simd_lane_id.
  public uint thread_index;
  public uint simdgroup_index;
  public uint simd_lane_id;
}

// (column, row) of the lane's first element in each 8 x 8 fragment; its second is adjacent in the row.
public int2 simdgroup_lane_coordinates(uint simd_lane_id) {
  let quad_index = int(simd_lane_id / 4);
  let lane_row = (quad_index & 4) + int((simd_lane_id / 2) % 4);
  let lane_col = (quad_index & 2) * 2 + int(simd_lane_id % 2) * 2;
  return int2(lane_col, lane_row);
}

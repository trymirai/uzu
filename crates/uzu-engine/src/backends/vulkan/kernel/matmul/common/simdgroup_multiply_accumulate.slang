import definitions;
import matmul.common.defines;

// Counterpart of Metal matmul/common/simdgroup_multiply_accumulate.h's SimdgroupMMA lane layout: lane l of a 32-lane
// simdgroup owns the two adjacent row elements at get_lane_coordinates(l) of each 8 x 8 fragment. The multiply itself is
// ThreadgroupTile.matmul's, as Vulkan exposes no simdgroup_matrix or cooperative matrix on this device.

// Metal's ThreadContext for the logical simdgroups.
public struct ThreadContext {
  // simdgroup_index * SIMD_SIZE + simd_lane_id, the local invocation index.
  public uint thread_index;
  public uint simdgroup_index;
  public uint simd_lane_id;
}

// The context of local invocation `tid`. Every invocation of the workgroup must call it, before any other use of
// threadgroup memory.
public ThreadContext simdgroup_thread_context(uint tid, uint simdgroups) {
  let native = subgroups_full32(simdgroups);
  ThreadContext context;
  context.simdgroup_index = native ? subgroup_id() : tid / SIMD_SIZE;
  context.simd_lane_id = native ? WaveGetLaneIndex() : tid % SIMD_SIZE;
  context.thread_index = context.simdgroup_index * SIMD_SIZE + context.simd_lane_id;
  return context;
}

// SimdgroupMMA::get_lane_coordinates: (column, row) of the lane's first element.
public int2 simdgroup_lane_coordinates(uint simd_lane_id) {
  let quad_index = int(simd_lane_id / 4);
  let lane_row = (quad_index & 4) + int((simd_lane_id / 2) % 4);
  let lane_col = (quad_index & 2) * 2 + int(simd_lane_id % 2) * 2;
  return int2(lane_col, lane_row);
}

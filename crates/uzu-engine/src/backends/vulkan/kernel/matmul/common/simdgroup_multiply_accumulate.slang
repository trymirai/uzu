import definitions;
import matmul.common.defines;

// Counterpart of Metal matmul/common/simdgroup_multiply_accumulate.h's SimdgroupMMA: lane l of a 32-lane simdgroup owns
// the two adjacent row elements at get_lane_coordinates(l) of each 8 x 8 fragment.

// Metal's ThreadContext for the logical simdgroups.
public struct ThreadContext {
  // simdgroup_index * SIMD_SIZE + simd_lane_id, the local invocation index.
  public uint thread_index;
  public uint simdgroup_index;
  public uint simd_lane_id;
  // Vulkan only: whether the logical simdgroups are the subgroups, so lanes may exchange fragments by shuffles.
  public bool native;
}

// The context of local invocation `tid`. Every invocation of the workgroup must call it, before any other use of
// threadgroup memory.
public ThreadContext simdgroup_thread_context(uint tid, uint simdgroups) {
  let native = subgroups_full32(simdgroups);
  ThreadContext context;
  context.simdgroup_index = native ? subgroup_id() : tid / SIMD_SIZE;
  context.simd_lane_id = native ? WaveGetLaneIndex() : tid % SIMD_SIZE;
  context.thread_index = context.simdgroup_index * SIMD_SIZE + context.simd_lane_id;
  context.native = native;
  return context;
}

// SimdgroupMMA::get_lane_coordinates: (column, row) of the lane's first element.
public int2 simdgroup_lane_coordinates(uint simd_lane_id) {
  let quad_index = int(simd_lane_id / 4);
  let lane_row = (quad_index & 4) + int((simd_lane_id / 2) % 4);
  let lane_col = (quad_index & 2) * 2 + int(simd_lane_id % 2) * 2;
  return int2(lane_col, lane_row);
}

// The inverse of simdgroup_lane_coordinates: the lane owning element (row, column) of a fragment.
uint simdgroup_lane(int row, int column) {
  return uint(((row & 4) << 2) | ((column & 4) << 1) | ((row & 3) << 1) | ((column & 2) >> 1));
}

// SimdgroupMMA::mma on native subgroup lanes: d = a b + c over 8 x 8 fragments, every lane folding
// fma(a[row][k], b[k][column + i], c[row][column + i]) for k ascending. b is first transposed across lanes so that each
// depth's two b elements of a lane come from a single lane; the elements of a and b reach a lane by shuffles.
public float2 simdgroup_mma(float2 a, float2 b, float2 c, uint simd_lane_id) {
  let coordinates = simdgroup_lane_coordinates(simd_lane_id);
  let row = coordinates.y;
  let column = coordinates.x;
  float2 transposed;
  [unroll]
  for (int i = 0; i < 2; ++i) {
    let owner = simdgroup_lane(column | i, row);
    let low = WaveReadLaneAt(b.x, owner);
    let high = WaveReadLaneAt(b.y, owner);
    transposed[i] = (row & 1) != 0 ? high : low;
  }
  var d = c;
  [unroll]
  for (int k = 0; k < 8; ++k) {
    let left = WaveReadLaneAt(a[k & 1], simdgroup_lane(row, k));
    [unroll]
    for (int i = 0; i < 2; ++i) {
      d[i] = fma(left, WaveReadLaneAt(transposed[k & 1], simdgroup_lane(column | i, k)), d[i]);
    }
  }
  return d;
}

// Vulkan only: the a and b elements of a workgroup's one logical simdgroup when its lanes are not subgroup lanes.
groupshared float4 simdgroup_mma_lanes[SIMD_SIZE];

// simdgroup_mma for a workgroup of one logical simdgroup whose lanes cannot shuffle: the same fma folds, the elements
// exchanged through shared memory, every invocation at both barriers.
public float2 simdgroup_mma_shared(float2 a, float2 b, float2 c, uint simd_lane_id) {
  simdgroup_mma_lanes[simd_lane_id] = float4(a, b);
  GroupMemoryBarrierWithGroupSync();
  let coordinates = simdgroup_lane_coordinates(simd_lane_id);
  var d = c;
  [unroll]
  for (int k = 0; k < 8; ++k) {
    let left = simdgroup_mma_lanes[simdgroup_lane(coordinates.y, k)][k & 1];
    [unroll]
    for (int i = 0; i < 2; ++i) {
      d[i] = fma(left, simdgroup_mma_lanes[simdgroup_lane(k, coordinates.x + i)][2 + i], d[i]);
    }
  }
  GroupMemoryBarrierWithGroupSync();
  return d;
}

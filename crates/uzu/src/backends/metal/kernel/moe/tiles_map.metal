#include <metal_stdlib>
#include <metal_atomic>
#include "../definitions.metal"

#define BM 16

// Compute per-expert tile counts: tiles_e = ceil((seg_len)/BM)
KERNEL(MoeTileCounts)(
    device const uint* offsets, // [e+1]
    device uint* tile_counts,   // [e]
    constant uint& e,
    const uint gid AXIS(e, 256)
) {
  const uint start = offsets[gid];
  const uint end = offsets[gid + 1];
  const uint seg_len = (end > start) ? (end - start) : 0u;
  const uint tiles = (seg_len == 0u) ? 0u : ((seg_len + BM - 1u) / BM);
  tile_counts[gid] = tiles;
}

// Single-threadgroup exclusive scan over tile_counts -> tile_row_offsets, also
// writes total_tiles
KERNEL(MoeTileScan)(
    device const uint* tile_counts, // [e]
    device uint* tile_row_offsets,  // [e+1]
    device uint* total_tiles_buf,   // [>=2]
    constant uint& e,
    threadgroup uint scratch[1024], // supports up to E<=1023
    uint tid THREADS(256)
) {
  if (tid < e) {
    scratch[tid] = tile_counts[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Inclusive scan in-place over scratch
  // Simple O(log E) scan; E is small
  for (uint offset = 1u; offset < e; offset <<= 1u) {
    uint val = 0u;
    if (tid < e) {
      val = scratch[tid];
      if (tid >= offset) {
        val += scratch[tid - offset];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < e) {
      scratch[tid] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u) {
    tile_row_offsets[0] = 0u;
  }
  if (tid < e) {
    // exclusive: shift right by one
    const uint prefix = (tid == 0u) ? 0u : scratch[tid - 1u];
    tile_row_offsets[tid] = prefix;
  }
  if (tid == 0u) {
    tile_row_offsets[e] = (e == 0u) ? 0u : scratch[e - 1u];
    total_tiles_buf[0] = tile_row_offsets[e];
    total_tiles_buf[1] = 0u;
    total_tiles_buf[2] = 0u;
    total_tiles_buf[3] = 0u;
    total_tiles_buf[4] = 0u;
    total_tiles_buf[5] = 0u;
    total_tiles_buf[6] = 0u;
    total_tiles_buf[7] = 0u;
  }
}

// Build flattened tile_map of length total_tiles; each entry is 3 uints:
// [3*i+0]=expert_idx, [3*i+1]=seg_start, [3*i+2]=tile_m0
KERNEL(MoeBuildTileMap)(
    device const uint* offsets,          // [e+1]
    device const uint* tile_row_offsets, // [e+1]
    device const uint* tile_counts,      // [e]
    device uint* tile_map,               // [3*total_tiles]
    constant uint& e,
    uint gid AXIS(e, 256)
) {
  const uint start = offsets[gid];
  const uint tiles = tile_counts[gid];
  const uint base = tile_row_offsets[gid];
  for (uint t = 0u; t < tiles; ++t) {
    const uint idx = base + t;
    tile_map[3u * idx + 0u] = gid;
    tile_map[3u * idx + 1u] = start;
    tile_map[3u * idx + 2u] = t * BM;
  }
}

// Write MTLDispatchThreadgroupsIndirectArguments {x, y, z} where:
//  x = num_tiles_n (computed on CPU and passed in), y = total_tiles, z = 1
KERNEL(MoeWriteDispatchArgs)(
    device const uint* total_tiles_buf, // [>=1], total_tiles_buf[0] = total_rows
    device uint* dispatch_args,         // [3] u32
    constant uint& num_tiles_n,         // x dimension
    uint gid AXIS(1, 1)
) {
  const uint total_rows = total_tiles_buf[0];
  dispatch_args[0] = num_tiles_n; // x
  dispatch_args[1] = total_rows;  // y
  dispatch_args[2] = 1u;          // z
}
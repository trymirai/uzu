#include <metal_stdlib>
#include "../definitions.metal"

// === Helper kernels for indirect dispatch of Pass A ===

// Count tiles per expert: tiles = (num_rows > 0) ? num_rows * h_blocks : 0
KERNEL(MoePassATileCounts)(
    device const uint* expert_offsets, // [e+1]
    device uint* tile_counts,          // [e]
    constant uint& e,
    constant uint& h_blocks,
    const uint tid AXIS(e, 256)
) {
  const uint start = expert_offsets[tid];
  const uint end = expert_offsets[tid + 1];
  const uint num_rows = end - start;
  tile_counts[tid] = (num_rows > 0) ? (num_rows * h_blocks) : 0;
}

// Exclusive scan of tile_counts to get tile_offsets and total_tiles
KERNEL(MoePassATileScan)(
    device const uint* tile_counts, // [e]
    device uint* tile_offsets,      // [e+1]
    device uint* total_tiles,       // [1]
    constant uint& e,
    threadgroup uint scratch[1024],
    uint gid GROUPS(1),
    uint lid THREADS(1024)
) {
  // Simple single-threadgroup scan (works for E <= 1024)
  const uint idx = lid;

  // Load into threadgroup memory
  if (idx < e) {
    scratch[idx] = tile_counts[idx];
  } else {
    scratch[idx] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Kogge-Stone scan
  uint val = scratch[idx];
  for (uint offset = 1; offset < 1024; offset *= 2) {
    uint temp = 0;
    if (idx >= offset && idx < e) {
      temp = scratch[idx - offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (idx >= offset && idx < e) {
      val += temp;
      scratch[idx] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write exclusive scan (shift right by 1)
  if (idx == 0) {
    tile_offsets[0] = 0;
  }
  if (idx < e) {
    tile_offsets[idx + 1] = scratch[idx];
    if (idx == e - 1) {
      total_tiles[0] = scratch[idx];
    }
  }
}

// Build row→expert map: one thread per routed row
KERNEL(MoePassABuildRowMap)(
    device const uint* expert_offsets, // [e+1]
    device uint* row_expert_map,       // [total_rows]
    constant uint& total_rows,
    constant uint& e,
    uint tid AXIS(total_rows, 256)
) {
  uint left = 0u;
  uint right = e;
  const uint row = tid;

  while (left + 1u < right) {
    const uint mid = (left + right) >> 1;
    if (row < expert_offsets[mid]) {
      right = mid;
    } else {
      left = mid;
    }
  }

  row_expert_map[row] = left;
}

// Build tile map entries from row→expert map
KERNEL(MoePassABuildTileMap)(
    device const uint* expert_offsets, // [E+1]
    device const uint* tile_offsets,   // [E+1]
    device const uint* row_expert_map, // [total_rows]
    device uint* tile_map,             // [total_tiles * 3]
    constant uint& total_rows,
    constant uint& h_blocks,
    uint tid AXIS(total_rows.saturating_mul(h_blocks), 256)
) {
  const uint row_idx = tid / h_blocks;
  const uint h_block = tid % h_blocks;
  if (row_idx >= total_rows) {
    return;
  }

  const uint expert_idx = row_expert_map[row_idx];
  const uint row_start = expert_offsets[expert_idx];
  const uint row_in_expert = row_idx - row_start;
  const uint tile_base =
      tile_offsets[expert_idx] + row_in_expert * h_blocks + h_block;

  tile_map[tile_base * 3 + 0] = h_block;
  tile_map[tile_base * 3 + 1] = expert_idx;
  tile_map[tile_base * 3 + 2] = row_in_expert;
}

// Write dispatch args for indirect dispatch (reusable from tiled version)
KERNEL(MoePassAWriteDispatchArgs)(
    device const uint* total_tiles, // [1]
    device uint* dispatch_args,     // [3] - MTLDispatchThreadgroupsIndirectArguments
    constant uint& num_tiles_y,     // usually 1 for Pass A
    uint tid AXIS(1, 1)
) {
  dispatch_args[0] = total_tiles[0]; // x dimension = total tiles
  dispatch_args[1] = num_tiles_y;    // y dimension
  dispatch_args[2] = 1;              // z dimension
}

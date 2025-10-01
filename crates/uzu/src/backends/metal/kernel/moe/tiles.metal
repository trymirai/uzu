#include <metal_stdlib>
using namespace metal;

#define BM 32

// Compute per-expert tile counts: tiles_e = ceil((seg_len)/BM)
kernel void moe_tile_counts(
    device const uint* offsets [[buffer(0)]],   // [E+1]
    device uint* tile_counts [[buffer(1)]],     // [E]
    constant uint& E [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= E) return;
    const uint start = offsets[gid];
    const uint end = offsets[gid + 1];
    const uint seg_len = (end > start) ? (end - start) : 0u;
    const uint tiles = (seg_len == 0u) ? 0u : ((seg_len + BM - 1u) / BM);
    tile_counts[gid] = tiles;
}

// Single-threadgroup exclusive scan over tile_counts -> tile_row_offsets, also writes total_tiles
kernel void moe_tile_scan(
    device const uint* tile_counts [[buffer(0)]],    // [E]
    device uint* tile_row_offsets [[buffer(1)]],     // [E+1]
    device uint* total_tiles_buf [[buffer(2)]],      // [1]
    constant uint& E [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint scratch[1024]; // supports up to E<=1023
    if (tid < E) {
        scratch[tid] = tile_counts[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inclusive scan in-place over scratch
    // Simple O(log E) scan; E is small
    for (uint offset = 1u; offset < E; offset <<= 1u) {
        uint val = 0u;
        if (tid < E) {
            val = scratch[tid];
            if (tid >= offset) {
                val += scratch[tid - offset];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < E) {
            scratch[tid] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        tile_row_offsets[0] = 0u;
    }
    if (tid < E) {
        // exclusive: shift right by one
        const uint prefix = (tid == 0u) ? 0u : scratch[tid - 1u];
        tile_row_offsets[tid] = prefix;
    }
    if (tid == 0u) {
        tile_row_offsets[E] = (E == 0u) ? 0u : scratch[E - 1u];
        total_tiles_buf[0] = tile_row_offsets[E];
    }
}

// Build flattened tile_map of length total_tiles; each entry is 3 uints:
// [3*i+0]=expert_idx, [3*i+1]=seg_start, [3*i+2]=tile_m0
kernel void moe_build_tile_map(
    device const uint* offsets [[buffer(0)]],           // [E+1]
    device const uint* tile_row_offsets [[buffer(1)]],  // [E+1]
    device const uint* tile_counts [[buffer(2)]],       // [E]
    device uint* tile_map [[buffer(3)]],                // [3*total_tiles]
    constant uint& E [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= E) return;
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



// Pass A: Per-threadgroup partial histograms in threadgroup memory (no atomics)
// Pass B: Reduce partials across blocks into final counts

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#define BLOCK_SIZE 256
#define SIMD_WIDTH 32
#define NUM_SG (BLOCK_SIZE / SIMD_WIDTH)
#define TILE_E 512

// Pass A: write partial histograms per block per tile (no device atomics)
kernel void moe_bucket_partials(
    device const int* topk_ids [[buffer(0)]],
    device uint* partials [[buffer(1)]],
    constant uint& T [[buffer(2)]],
    constant uint& E [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& num_blocks [[buffer(5)]],
    uint3 tid3 [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    if (E == 0) return;

    const uint gid = tid3.x;
    const uint sg_id = lid / SIMD_WIDTH;
    const uint lane  = lid % SIMD_WIDTH;
    const uint grid_stride = num_blocks * BLOCK_SIZE;

    threadgroup _atomic<uint> tg_hist[TILE_E];

    // Tile E dimension
    for (uint e0 = 0; e0 < E; e0 += TILE_E) {
        const uint tile_e = (e0 + TILE_E <= E) ? TILE_E : (E - e0);

        // Zero threadgroup histogram
        for (uint e = lid; e < tile_e; e += BLOCK_SIZE) {
            atomic_store_explicit(&tg_hist[e], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Grid-stride over tokens
        for (uint i = gid; i < T; i += grid_stride) {
            const uint base = i * K;
            for (uint k = 0; k < K; ++k) {
                int eid = topk_ids[base + k];
                if (eid >= 0) {
                    uint ue = uint(eid);
                    if (ue >= e0 && ue < e0 + tile_e) {
                        uint te = ue - e0;
                        atomic_fetch_add_explicit(&tg_hist[te], 1u, memory_order_relaxed);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Store partials for this block and tile (contiguous, stride TILE_E)
        const uint block_id = tgpig.x;
        const uint num_tiles = (E + TILE_E - 1u) / TILE_E;
        const uint tile_id = e0 / TILE_E;
        const uint base_idx = (block_id * num_tiles + tile_id) * TILE_E;
        for (uint e = lid; e < tile_e; e += BLOCK_SIZE) {
            partials[base_idx + e] = atomic_load_explicit(&tg_hist[e], memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Pass B: reduce partial histograms across blocks
kernel void moe_bucket_reduce_partials(
    device const uint* partials [[buffer(0)]],
    device uint* counts [[buffer(1)]],
    constant uint& E [[buffer(2)]],
    constant uint& num_blocks [[buffer(3)]],
    uint3 tid3 [[thread_position_in_grid]])
{
    const uint num_tiles = (E + TILE_E - 1u) / TILE_E;
    const uint e = tid3.x;
    if (e >= E) return;
    const uint tile_id = e / TILE_E;
    const uint te = e % TILE_E;
    uint sum = 0u;
    for (uint b = 0; b < num_blocks; ++b) {
        const uint base_idx = (b * num_tiles + tile_id) * TILE_E;
        sum += partials[base_idx + te];
    }
    counts[e] = sum;
}



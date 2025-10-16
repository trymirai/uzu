#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#include "../definitions.metal"

#define BLOCK_SIZE 128
#define TILE_E 512

// Single-kernel fused: count all experts + scan to offsets
// This kernel is launched with SINGLE threadgroup
kernel void moe_counts_offsets_fused(
    device const int* topk_ids [[buffer(0)]],
    device uint* offsets [[buffer(1)]],          // output: exclusive scan [E+1]
    device uint* sum_k_out [[buffer(2)]],        // output: total count [1]
    device uint* partials [[buffer(3)]],         // output: partials [num_tiles * TILE_E] (for block_bases)
    constant uint& T [[buffer(4)]],
    constant uint& E [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint lid [[thread_index_in_threadgroup]])
{
    if (E == 0) {
        if (lid == 0) {
            offsets[0] = 0u;
            sum_k_out[0] = 0u;
        }
        return;
    }

    threadgroup _atomic<uint> tg_hist[TILE_E];
    threadgroup uint scan_shared[BLOCK_SIZE];
    threadgroup uint reduce_shared[BLOCK_SIZE];
    threadgroup uint counts_shared[BLOCK_SIZE];  // Cache counts in threadgroup memory

    // ═══════════════════════════════════════════════════════════
    // PHASE 1: Count tokens per expert using tiled histogram
    // ═══════════════════════════════════════════════════════════
    // Tile over E dimension
    for (uint e0 = 0; e0 < E; e0 += TILE_E) {
        const uint tile_e = (e0 + TILE_E <= E) ? TILE_E : (E - e0);

        // Zero threadgroup histogram
        for (uint e = lid; e < tile_e; e += BLOCK_SIZE) {
            atomic_store_explicit(&tg_hist[e], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process all tokens in strided fashion
        // Since we have only 1 threadgroup, we need to cover all T tokens
        for (uint t = lid; t < T; t += BLOCK_SIZE) {
            const uint base = t * K;
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

        // Write final counts and partials for this tile
        for (uint e = lid; e < tile_e; e += BLOCK_SIZE) {
            const uint count_val = atomic_load_explicit(&tg_hist[e], memory_order_relaxed);
            counts_shared[e0 + e] = count_val;
            partials[e0 + e] = count_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ═══════════════════════════════════════════════════════════
    // PHASE 2: Compute exclusive prefix scan on counts
    // ═══════════════════════════════════════════════════════════
    threadgroup uint carry;
    if (lid == 0) { carry = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0; base < E; base += BLOCK_SIZE) {
        uint remaining = E - base;
        uint chunk_n = remaining < BLOCK_SIZE ? remaining : BLOCK_SIZE;

        uint v = (lid < chunk_n) ? counts_shared[lid] : 0u;

        uint prefix_local = threadgroup_raking_prefix_exclusive_sum<BLOCK_SIZE>(v, scan_shared, (ushort)lid);
        uint prefix_global = prefix_local + carry;
        if (lid < chunk_n) {
            offsets[base + lid] = prefix_global;
        }

        uint block_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(v, reduce_shared, (ushort)lid);
        if (lid == 0) { carry += block_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final offset and total
    if (lid == 0) {
        offsets[E] = carry;
        sum_k_out[0] = carry;
    }
}

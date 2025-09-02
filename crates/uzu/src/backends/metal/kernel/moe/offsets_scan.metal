#include <metal_stdlib>
using namespace metal;

#include "../definitions.metal"

#define BLOCK_SIZE 256

kernel void moe_offsets_exclusive_scan(
    device const uint* counts [[buffer(0)]],
    device uint* offsets [[buffer(1)]],      // length E+1
    device uint* sum_k_out [[buffer(2)]],    // length 1
    constant uint& E [[buffer(3)]],
    uint lid [[thread_index_in_threadgroup]])
{
    if (E == 0) {
        if (lid == 0) {
            offsets[0] = 0u;
            sum_k_out[0] = 0u;
        }
        return;
    }

    threadgroup uint scan_shared[BLOCK_SIZE];
    threadgroup uint red_shared[BLOCK_SIZE];
    threadgroup uint carry;
    if (lid == 0) { carry = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0; base < E; base += BLOCK_SIZE) {
        uint remaining = E - base;
        uint chunk_n = remaining < BLOCK_SIZE ? remaining : BLOCK_SIZE;

        uint v = (lid < chunk_n) ? counts[base + lid] : 0u;

        uint prefix_local = threadgroup_raking_prefix_exclusive_sum<BLOCK_SIZE>(v, scan_shared, (ushort)lid);
        uint prefix_global = prefix_local + carry;
        if (lid < chunk_n) {
            offsets[base + lid] = prefix_global;
        }

        uint block_sum = threadgroup_raking_reduce_sum<BLOCK_SIZE>(v, red_shared, (ushort)lid);
        if (lid == 0) { carry += block_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        offsets[E] = carry;
        sum_k_out[0] = carry;
    }
}



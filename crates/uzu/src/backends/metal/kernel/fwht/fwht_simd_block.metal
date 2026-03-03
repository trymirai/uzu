#include <metal_stdlib>
#include "../definitions.metal"

#include "fwht.h"

// Hybrid approach: preload into threadgroup memory, then use simd_shuffle
// for the butterfly computation. Each SIMD group (32 lanes) processes one
// 32-element sub-block independently.

template <typename T, int N>
VARIANTS(T, half, float, bfloat)
VARIANTS(N, 32, 64, 128)
KERNEL(FwhtSimdBlock)(
    device T* data,
    constant uint& batch_size,
    constant float& scale,
    threadgroup T shared_buf[N],
    const uint group_idx GROUPS(batch_size),
    const uint tid THREADS(128)
) {
    if (tid >= N) return;

    device T* block = data + group_idx * N;

    shared_buf[tid] = block[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float val = float(shared_buf[tid]);
    ushort lane = tid & 31;

    STEEL_PRAGMA_UNROLL
    for (ushort h = 1; h < 32; h <<= 1) {
        float other = simd_shuffle_xor(val, h);
        val = (lane & h) ? (other - val) : (val + other);
    }

    block[tid] = T(val * scale);
}

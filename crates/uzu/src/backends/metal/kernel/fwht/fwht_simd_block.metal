#include <metal_stdlib>
#include "../definitions.metal"

#include "fwht.h"

// Preload into threadgroup memory, then use simd_shuffle for the butterfly.
// Each SIMD group (32 lanes) processes one 32-element sub-block independently.

template <typename T, int N>
VARIANTS(T, half, float, bfloat)
VARIANTS(N, 128)
KERNEL(FwhtSimdBlock)(
    device T* data,
    constant uint& batch_size,
    constant float& scale,
    threadgroup T threadgroup_buffer[N],
    const uint group_index GROUPS(batch_size),
    const uint thread_index THREADS(128)
) {
    if (thread_index >= N) return;

    device T* block_start = data + group_index * N;

    threadgroup_buffer[thread_index] = block_start[thread_index];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float value = float(threadgroup_buffer[thread_index]);
    ushort lane_index = thread_index & 31;

    STEEL_PRAGMA_UNROLL
    for (ushort butterfly_stride = 1; butterfly_stride < 32; butterfly_stride <<= 1) {
        float partner_value = simd_shuffle_xor(value, butterfly_stride);
        value = (lane_index & butterfly_stride) ? (partner_value - value) : (value + partner_value);
    }

    block_start[thread_index] = T(value * scale);
}

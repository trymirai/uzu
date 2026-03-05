#include <metal_stdlib>
#include "../definitions.metal"

#include "fwht.h"

template <typename T> struct compute_type_of { using type = float; };
template <> struct compute_type_of<half> { using type = half; };
template <> struct compute_type_of<char> { using type = int; };

template <typename T, int N>
VARIANTS(T, half, float, bfloat, char)
VARIANTS(N, 128, 512, 1024, 2048)
KERNEL(FwhtSimdBlock)(
    device T* data,
    constant uint& batch_size,
    constant float& scale,
    const uint group_index GROUPS(batch_size),
    const uint thread_index THREADS(128)
) {
    if (thread_index >= 128) return;

    using C = typename compute_type_of<T>::type;

    constexpr short elements_per_thread = N / 128;
    constexpr short sub_block_size = 32 * elements_per_thread;

    device T* block_start = data + group_index * N;
    ushort lane_index = thread_index & 31;
    ushort sub_block_offset = (thread_index / 32) * sub_block_size;

    C reg[elements_per_thread];
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < elements_per_thread; r++) {
        reg[r] = C(block_start[sub_block_offset + lane_index + 32 * r]);
    }

    IF_CONSTEXPR(elements_per_thread > 1) {
        radix_func<elements_per_thread, C>(reg);
    }

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < elements_per_thread; r++) {
        STEEL_PRAGMA_UNROLL
        for (ushort s = 1; s < 32; s *= 2) {
            C partner = simd_shuffle_xor(reg[r], s);
            reg[r] = (lane_index & s) ? (partner - reg[r]) : (reg[r] + partner);
        }
    }

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < elements_per_thread; r++) {
        block_start[sub_block_offset + lane_index + 32 * r] = T(float(reg[r]) * scale);
    }
}

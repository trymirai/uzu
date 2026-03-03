#include <metal_stdlib>
#include "../definitions.metal"

#include "fwht.h"

// Full-vector in-place Hadamard transform.
// Ported from MLX's hadamard_n kernel.
// One threadgroup processes one row of length N.
// Uses threadgroup memory for inter-thread communication.

#define MAX_RADIX 16

template <typename T, int N>
VARIANTS(T, half, float, bfloat)
VARIANTS(N, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
KERNEL(Fwht)(
    device T* data,
    constant uint& batch_size,
    constant float& scale,
    threadgroup T threadgroup_buffer[N],
    const uint batch_index GROUPS(batch_size),
    const uint thread_index THREADS(512)
) {
    constexpr short max_radix = MAX_RADIX;
    constexpr short num_threads = N / max_radix;
    if (thread_index >= num_threads) return;
    constexpr short log_n = __builtin_ctz(N);
    constexpr short log_radix = __builtin_ctz(max_radix);
    constexpr short num_stages = log_n / log_radix;
    constexpr short log_final_radix = log_n % log_radix;
    constexpr short final_radix = 1 << log_final_radix;

    device T* row = data + batch_index * N;
    short thread_id = thread_index;

    STEEL_PRAGMA_UNROLL
    for (short element_index = 0; element_index < max_radix; element_index++) {
        threadgroup_buffer[element_index * num_threads + thread_id] = row[element_index * num_threads + thread_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float register_buffer[max_radix];
    short butterfly_stride = 1;

    STEEL_PRAGMA_UNROLL
    for (short stage = 0; stage < num_stages; stage++) {
        short low_bits = thread_id & (butterfly_stride - 1);
        short base_index = ((thread_id - low_bits) << log_radix) + low_bits;

        STEEL_PRAGMA_UNROLL
        for (short radix_index = 0; radix_index < max_radix; radix_index++) {
            register_buffer[radix_index] = float(threadgroup_buffer[base_index + butterfly_stride * radix_index]);
        }

        radix_func<max_radix>(register_buffer);

        STEEL_PRAGMA_UNROLL
        for (short radix_index = 0; radix_index < max_radix; radix_index++) {
            threadgroup_buffer[base_index + butterfly_stride * radix_index] = T(register_buffer[radix_index]);
        }

        butterfly_stride <<= log_radix;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    IF_CONSTEXPR(final_radix > 1) {
        STEEL_PRAGMA_UNROLL
        for (int sub_block = 0; sub_block < max_radix / final_radix; sub_block++) {
            short expanded_index = thread_id + sub_block * num_threads;
            short low_bits = expanded_index & (butterfly_stride - 1);
            short base_index = ((expanded_index - low_bits) << log_final_radix) + low_bits;

            STEEL_PRAGMA_UNROLL
            for (short radix_index = 0; radix_index < final_radix; radix_index++) {
                register_buffer[radix_index] = float(threadgroup_buffer[base_index + butterfly_stride * radix_index]);
            }

            radix_func<final_radix>(register_buffer);

            STEEL_PRAGMA_UNROLL
            for (short radix_index = 0; radix_index < final_radix; radix_index++) {
                threadgroup_buffer[base_index + butterfly_stride * radix_index] = T(register_buffer[radix_index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (short element_index = 0; element_index < max_radix; element_index++) {
        row[element_index * num_threads + thread_id] =
            T(float(threadgroup_buffer[element_index * num_threads + thread_id]) * scale);
    }
}

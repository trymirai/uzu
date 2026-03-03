#include <metal_stdlib>
#include "../definitions.metal"

#include "fwht.h"
#include "fwht_m.h"

// Dense Hadamard transform for the non-power-of-two factor M in {12, 20, 28}.
//
// Data layout: each row has M * power_of_two_size contiguous elements,
// viewed as M groups of power_of_two_size.
// Each thread applies the O(M^2) codelet to one column position, loading
// M values strided by power_of_two_size, transforming, and writing back.

template <typename T, int M>
VARIANTS(T, half, float, bfloat)
VARIANTS(M, 12, 20, 28)
KERNEL(FwhtM)(
    device T* data,
    constant uint& batch_size,
    constant uint& n,
    constant float& scale,
    const uint group_index GROUPS(batch_size),
    const uint thread_index THREADS(256)
) {
    device T* row = data + group_index * M * n;

    for (uint column_position = thread_index; column_position < n; column_position += 256) {
        float column_values[M];

        STEEL_PRAGMA_UNROLL
        for (short m_index = 0; m_index < M; m_index++) {
            column_values[m_index] = float(row[m_index * n + column_position]);
        }

        IF_CONSTEXPR(M == 12) { hadamard_radix_12(column_values); }
        IF_CONSTEXPR(M == 20) { hadamard_radix_20(column_values); }
        IF_CONSTEXPR(M == 28) { hadamard_radix_28(column_values); }

        STEEL_PRAGMA_UNROLL
        for (short m_index = 0; m_index < M; m_index++) {
            row[m_index * n + column_position] = T(column_values[m_index] * scale);
        }
    }
}

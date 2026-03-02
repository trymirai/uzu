#include <metal_stdlib>
#include "../definitions.metal"

#include "fwht.h"
#include "fwht_m.h"

// Dense Hadamard transform for the non-power-of-two factor m in {12, 20, 28}.
//
// Data layout: each row has m*n contiguous elements, viewed as m groups of n.
// Each thread applies the O(m^2) codelet to one column position, loading
// m values strided by n, transforming, and writing back.

template <typename T, int M>
VARIANTS(T, half, float, bfloat)
VARIANTS(M, 12, 20, 28)
KERNEL(FwhtM)(
    device T* data,
    constant uint& batch_size,
    constant uint& n,
    constant float& scale,
    const uint group_idx GROUPS(batch_size),
    const uint tid THREADS(256)
) {
  device T* row = data + group_idx * M * n;

  for (uint pos = tid; pos < n; pos += 256) {
    float x[M];

    STEEL_PRAGMA_UNROLL
    for (short c = 0; c < M; c++) {
      x[c] = float(row[c * n + pos]);
    }

    IF_CONSTEXPR(M == 12) { hadamard_radix_12(x); }
    IF_CONSTEXPR(M == 20) { hadamard_radix_20(x); }
    IF_CONSTEXPR(M == 28) { hadamard_radix_28(x); }

    STEEL_PRAGMA_UNROLL
    for (short c = 0; c < M; c++) {
      row[c * n + pos] = T(x[c] * scale);
    }
  }
}

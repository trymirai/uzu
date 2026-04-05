#include <metal_stdlib>
#include "../common/dsl.h"
#include "quant_matmul.h"

// Small-tile QMM for speculative decoding verification (M=2-31).
// BM=8, BK=32, BN=32, WM=1, WN=1 → 1 simdgroup, 32 threads.
// Weights dequantized to threadgroup memory once,
// shared across M rows via hardware simdgroup_multiply_accumulate.
// BM=8 minimizes padding waste for M=2-8 (typical spec decode).
template <typename T, int GROUP_SIZE, int BITS>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
PUBLIC KERNEL(QuantizedMatmulQmmTransposedSmall)(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* x,
    device T* y,
    const constant int& k,
    const constant int& n,
    const constant int& m,
    threadgroup T Xs[8 * (32 + 16 / sizeof(T))],
    threadgroup T Ws[32 * (32 + 16 / sizeof(T))],
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const uint tgid_x GROUPS((n + 32 - 1) / 32),
    const uint tgid_y GROUPS((m + 8 - 1) / 8),
    const uint tgid_z GROUPS(1),
    const uint tid_x THREADS(32),
    const uint tid_y THREADS(1),
    const uint tid_z THREADS(1)
) {
  const uint3 tid = uint3(tgid_x, tgid_y, tgid_z);
  const uint lid = tid_x;
  const uint simd_gid = 0;
  const uint simd_lid = tid_x;

  if (use_mlx_quant) {
    qmm_transposed_impl<T, GROUP_SIZE, BITS, true, 8, 32, 32, true, 1, 1>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        Xs,
        Ws,
        k,
        n,
        m,
        tid,
        lid,
        simd_gid,
        simd_lid
    );
  } else {
    qmm_transposed_impl<T, GROUP_SIZE, BITS, true, 8, 32, 32, false, 1, 1>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        Xs,
        Ws,
        k,
        n,
        m,
        tid,
        lid,
        simd_gid,
        simd_lid
    );
  }
}

#include <metal_stdlib>
#include "../definitions.metal"
#include "common.h"

template <typename T, int group_size, int bits>
VARIANTS(T, bfloat)
VARIANTS(group_size, 64, 128)
VARIANTS(bits, 4, 8)
KERNEL(QuantizedMatmulQmmTransposed64x64)(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    threadgroup T Xs[64 * (64 + 16 / sizeof(T))],
    threadgroup T Ws[64 * (64 + 16 / sizeof(T))],
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const uint tgid_x GROUPS((N + 64 - 1) / 64),
    const uint tgid_y GROUPS((M + 64 - 1) / 64),
    const uint tgid_z GROUPS(1),
    const uint tid_x THREADS(32),
    const uint tid_y THREADS(2),
    const uint tid_z THREADS(2)
) {
  const uint3 tid = uint3(tgid_x, tgid_y, tgid_z);
  const uint lid = tid_z * 64 + tid_y * 32 + tid_x;
  const uint simd_gid = tid_z * 2 + tid_y;
  const uint simd_lid = tid_x;

  if (use_mlx_quant) {
    qmm_transposed_impl<T, group_size, bits, true, 64, 64, 64, true>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        Xs,
        Ws,
        K,
        N,
        M,
        tid,
        lid,
        simd_gid,
        simd_lid
    );
  } else {
    qmm_transposed_impl<T, group_size, bits, true, 64, 64, 64, false>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        Xs,
        Ws,
        K,
        N,
        M,
        tid,
        lid,
        simd_gid,
        simd_lid
    );
  }
}

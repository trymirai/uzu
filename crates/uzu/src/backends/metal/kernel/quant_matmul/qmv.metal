#include <metal_stdlib>
#include "../definitions.metal"
#include "common.h"

template <typename T, int group_size, int bits>
VARIANTS(T, float, half, bfloat)
VARIANTS(group_size, 32, 64, 128)
VARIANTS(bits, 4, 8)
KERNEL(QuantizedMatmulQmv)(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const uint tgid_x GROUPS(M),
    const uint tgid_y GROUPS((N + 8 - 1) / 8),
    const uint tgid_z GROUPS(1),
    const uint tid_x THREADS(32),
    const uint tid_y THREADS(2)
) {
  const uint3 tid = uint3(tgid_x, tgid_y, tgid_z);
  const uint simd_gid = tid_y;
  const uint simd_lid = tid_x;

  if (use_mlx_quant) {
    qmv_impl<T, group_size, bits, true>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        K,
        N,
        tid,
        simd_gid,
        simd_lid
    );
  } else {
    qmv_impl<T, group_size, bits, false>(
        w,
        scales,
        zero_points,
        biases,
        x,
        y,
        K,
        N,
        tid,
        simd_gid,
        simd_lid
    );
  }
}

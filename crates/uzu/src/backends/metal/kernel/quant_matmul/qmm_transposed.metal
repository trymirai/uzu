#include <metal_stdlib>
#include "../definitions.metal"
#include "quant_matmul.metal"

template <typename T, int group_size, int bits>
VARIANTS(T, float, half, bfloat)
VARIANTS(group_size, 32, 64, 128)
VARIANTS(bits, 4, 8)
KERNEL(QuantizedMatmulQmmTransposed)(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* x,
    device T* y,
    const constant int& k,
    const constant int& n,
    const constant int& m,
    threadgroup T Xs[32 * (32 + 16 / sizeof(T))],
    threadgroup T Ws[32 * (32 + 16 / sizeof(T))],
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const bool aligned_n SPECIALIZE,
    const uint tgid_x GROUPS((n + 32 - 1) / 32),
    const uint tgid_y GROUPS((m + 32 - 1) / 32),
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
    if (aligned_n) {
      qmm_transposed_impl<T, group_size, bits, true, 32, 32, 32, true>(
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
      qmm_transposed_impl<T, group_size, bits, false, 32, 32, 32, true>(
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
  } else {
    if (aligned_n) {
      qmm_transposed_impl<T, group_size, bits, true, 32, 32, 32, false>(
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
      qmm_transposed_impl<T, group_size, bits, false, 32, 32, 32, false>(
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
}

#include <metal_stdlib>
#include "../common/dsl.h"
#include "quant_matmul.metal"

using namespace metal;

// Fused QmmTransposed + output Hadamard transform for prefill (batch >= 32).
// Phase 1: standard tiled quantized matmul writes results to device y.
// Phase 2: each simdgroup reads back its rows, applies the Hadamard butterfly
//          via simd_shuffle_xor, and overwrites y with the transformed output.
// Requires n % 32 == 0 (aligned_n = true) so every column tile is a full
// 32-element Hadamard block.

template <typename T, int GROUP_SIZE, int BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
PUBLIC KERNEL(QuantizedMatmulQmmTransposedOutputHadamard)(
    const device uint32_t* w,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* x,
    device T* y,
    const device T* output_factors,
    const constant int& k,
    const constant int& n,
    const constant int& m,
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    threadgroup T Xs[32 * (32 + 16 / sizeof(T))],
    threadgroup T Ws[32 * (32 + 16 / sizeof(T))],
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

  // ── Phase 1: Quantized MatMul (writes tile to device y) ──────────────
  if (use_mlx_quant) {
    qmm_transposed_impl<T, GROUP_SIZE, BITS, true, 32, 32, 32, true>(
        w, scales, zero_points, biases, x, y, Xs, Ws,
        k, n, m, tid, lid, simd_gid, simd_lid);
  } else {
    qmm_transposed_impl<T, GROUP_SIZE, BITS, true, 32, 32, 32, false>(
        w, scales, zero_points, biases, x, y, Xs, Ws,
        k, n, m, tid, lid, simd_gid, simd_lid);
  }

  // ── Phase 2: Ensure QMM device writes are visible to all threads ─────
  threadgroup_barrier(mem_flags::mem_device);

  // ── Phase 3: Hadamard transform on BM×BN tile ───────────────────────
  constexpr int BM = 32;
  constexpr int ROWS_PER_SG = BM / 4;
  const int y_row_base = tgid_y * BM;
  const int y_col_base = tgid_x * BM;

  for (int r = 0; r < ROWS_PER_SG; r++) {
    const int local_row = simd_gid * ROWS_PER_SG + r;
    const int global_row = y_row_base + local_row;
    const int global_col = y_col_base + simd_lid;

    float value = 0.0f;
    const bool valid = (global_row < m) && (global_col < n);

    if (valid) {
      value = float(y[global_row * n + global_col]);
      value *= float(output_factors[global_col]);
    }

    for (uint stride = 1; stride < METAL_SIMD_SIZE; stride <<= 1) {
      float partner = simd_shuffle_xor(value, static_cast<ushort>(stride));
      value = (simd_lid & stride) ? (partner - value) : (partner + value);
    }

    if (valid) {
      constexpr float normalization_factor = 1.0f / 5.656854249f;
      y[global_row * n + global_col] = T(value * normalization_factor);
    }
  }
}

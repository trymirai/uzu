#include <metal_stdlib>
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "quant_matmul.h"

// Transposed QMM kernel. BM = batch tile rows, BK = K-axis tile, BN = output
// tile cols. WM, WN = simdgroup tile counts along M and N axes.
template <typename T, uint GROUP_SIZE, uint BITS, uint BM, uint BK, uint BN, uint WM, uint WN>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
VARIANTS(BM, 8, 32, 64)
VARIANTS(BK, 32, 64)
VARIANTS(BN, 32, 64)
VARIANTS(WM, 1, 2)
VARIANTS(WN, 1, 2)
CONSTRAINT(
  (BM == 8  && BK == 32 && BN == 32 && WM == 1 && WN == 1) ||
  (BM == 32 && BK == 32 && BN == 32 && WM == 2 && WN == 2) ||
  (BM == 32 && BK == 64 && BN == 32 && WM == 2 && WN == 2) ||
  (BM == 64 && BK == 32 && BN == 64 && WM == 2 && WN == 2) ||
  (BM == 64 && BK == 64 && BN == 64 && WM == 2 && WN == 2))
CONSTRAINT(BK <= GROUP_SIZE)
CONSTRAINT(T != "float" || BK < 64)
PUBLIC KERNEL(QuantizedMatmulQmmTransposed)(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* input,
    device T* output,
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup T Xs[BM * (BK + 16 / sizeof(T))],
    threadgroup T Ws[BN * (BK + 16 / sizeof(T))],
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const bool aligned_n SPECIALIZE,
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(BN)),
    const uint batch_block_idx GROUPS(batch_size.div_ceil(BM)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(WM * WN)
) {
  if (use_mlx_quant) {
    if (aligned_n) {
      qmm_transposed_impl<T, GROUP_SIZE, BITS, true, BM, BK, BN, true, WM, WN>(
          weights,
          scales,
          zero_points,
          biases,
          input,
          output,
          Xs,
          Ws,
          in_vec_size,
          out_vec_size,
          batch_size,
          out_block_idx,
          batch_block_idx,
          simd_group,
          simd_lane
      );
    } else {
      qmm_transposed_impl<T, GROUP_SIZE, BITS, false, BM, BK, BN, true, WM, WN>(
          weights,
          scales,
          zero_points,
          biases,
          input,
          output,
          Xs,
          Ws,
          in_vec_size,
          out_vec_size,
          batch_size,
          out_block_idx,
          batch_block_idx,
          simd_group,
          simd_lane
      );
    }
  } else {
    if (aligned_n) {
      qmm_transposed_impl<T, GROUP_SIZE, BITS, true, BM, BK, BN, false, WM, WN>(
          weights,
          scales,
          zero_points,
          biases,
          input,
          output,
          Xs,
          Ws,
          in_vec_size,
          out_vec_size,
          batch_size,
          out_block_idx,
          batch_block_idx,
          simd_group,
          simd_lane
      );
    } else {
      qmm_transposed_impl<
          T,
          GROUP_SIZE,
          BITS,
          false,
          BM,
          BK,
          BN,
          false,
          WM,
          WN>(
          weights,
          scales,
          zero_points,
          biases,
          input,
          output,
          Xs,
          Ws,
          in_vec_size,
          out_vec_size,
          batch_size,
          out_block_idx,
          batch_block_idx,
          simd_group,
          simd_lane
      );
    }
  }

  if (use_hadamard) {
    threadgroup_barrier(mem_flags::mem_device);

    // Each lane covers BN/32 output columns (1 at BN=32, 2 at BN=64).
    constexpr uint NUM_SIMD_GROUPS = WM * WN;
    constexpr uint ROWS_PER_SIMD_GROUP = BM / NUM_SIMD_GROUPS;
    constexpr uint COLS_PER_LANE = BN / 32;

    const uint row_start =
        batch_block_idx * BM + simd_group * ROWS_PER_SIMD_GROUP;
    const uint row_end = min(row_start + ROWS_PER_SIMD_GROUP, batch_size);

    for (uint c = 0; c < COLS_PER_LANE; c++) {
      const uint out_col = out_block_idx * BN + c * 32 + simd_lane;
      const int32_t factor = hadamard_factors[out_col];

      for (uint row = row_start; row < row_end; row++) {
        uint output_index = row * out_vec_size + out_col;

        output[output_index] = simdgroup_random_hadamard_transform(
            static_cast<ushort>(simd_lane),
            output[output_index],
            factor
        );
      }
    }
  }
}

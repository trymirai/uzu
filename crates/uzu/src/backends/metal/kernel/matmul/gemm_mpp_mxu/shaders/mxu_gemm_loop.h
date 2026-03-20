#pragma once

#include "mxu_matmul.h"
#include "../../../../generated/matmul.h"

namespace uzu {
namespace matmul {

METAL_CONST short MXU_K_ITERATION_STRIDE = 256;
METAL_CONST short MXU_SIMDGROUP_K_STEP = 32;

template <
    typename T,
    short SIMDGROUP_ROWS,
    short SIMDGROUP_COLS,
    bool transpose_a,
    bool transpose_b,
    bool aligned_m,
    bool aligned_n,
    bool aligned_k,
    typename AccumulatorType = float>
METAL_FUNC MxuTile<AccumulatorType, SIMDGROUP_ROWS / 16, SIMDGROUP_COLS / 16>
mxu_gemm_loop(
    const device T* A,
    const device T* B,
    int leading_dimension_a,
    int leading_dimension_b,
    int K,
    const short simdgroup_limit_m,
    const short simdgroup_limit_n
) {
  constexpr short TILE_ROWS = SIMDGROUP_ROWS / 16;
  constexpr short TILE_COLS = SIMDGROUP_COLS / 16;
  constexpr short TILE_K = MXU_SIMDGROUP_K_STEP / 16;

  constexpr int A_TILE_ROWS = transpose_a ? TILE_K : TILE_ROWS;
  constexpr int A_TILE_COLS = transpose_a ? TILE_ROWS : TILE_K;

  constexpr int B_TILE_ROWS = transpose_b ? TILE_COLS : TILE_K;
  constexpr int B_TILE_COLS = transpose_b ? TILE_K : TILE_COLS;

  MxuTile<AccumulatorType, TILE_ROWS, TILE_COLS> accumulator;
  accumulator.clear();

  const int aligned_k_iterations = K / MXU_K_ITERATION_STRIDE;

  for (int outer_k = 0; outer_k < aligned_k_iterations; outer_k++) {
    for (int inner_k = 0; inner_k < MXU_K_ITERATION_STRIDE;
         inner_k += MXU_SIMDGROUP_K_STEP) {
      MxuTile<T, A_TILE_ROWS, A_TILE_COLS> a_tile;
      MxuTile<T, B_TILE_ROWS, B_TILE_COLS> b_tile;

      const int a_offset =
          transpose_a ? inner_k * leading_dimension_a : inner_k;
      const int b_offset =
          transpose_b ? inner_k : inner_k * leading_dimension_b;

      if constexpr (aligned_m) {
        a_tile.load(A + a_offset, leading_dimension_a);
      } else {
        const short row_max =
            transpose_a ? MXU_SIMDGROUP_K_STEP : simdgroup_limit_m;
        const short col_max =
            transpose_a ? simdgroup_limit_m : MXU_SIMDGROUP_K_STEP;
        a_tile.load_checked(
            A + a_offset,
            leading_dimension_a,
            short2(col_max, row_max)
        );
      }

      if constexpr (aligned_n) {
        b_tile.load(B + b_offset, leading_dimension_b);
      } else {
        const short row_max =
            transpose_b ? simdgroup_limit_n : MXU_SIMDGROUP_K_STEP;
        const short col_max =
            transpose_b ? MXU_SIMDGROUP_K_STEP : simdgroup_limit_n;
        b_tile.load_checked(
            B + b_offset,
            leading_dimension_b,
            short2(col_max, row_max)
        );
      }

      mxu_tiled_matmul(
          accumulator,
          a_tile,
          metal::bool_constant<transpose_a>{},
          b_tile,
          metal::bool_constant<transpose_b>{}
      );
    }

    A += transpose_a ? (MXU_K_ITERATION_STRIDE * leading_dimension_a)
                     : MXU_K_ITERATION_STRIDE;
    B += transpose_b ? MXU_K_ITERATION_STRIDE
                     : (MXU_K_ITERATION_STRIDE * leading_dimension_b);
  }

  if constexpr (!aligned_k) {
    const short remaining_k = K - aligned_k_iterations * MXU_K_ITERATION_STRIDE;

    for (int inner_k = 0; inner_k < remaining_k;
         inner_k += MXU_SIMDGROUP_K_STEP) {
      MxuTile<T, A_TILE_ROWS, A_TILE_COLS> a_tile;
      MxuTile<T, B_TILE_ROWS, B_TILE_COLS> b_tile;

      const short partial_k = max(0, remaining_k - inner_k);

      const short2 a_limits = transpose_a
                                  ? short2(simdgroup_limit_m, partial_k)
                                  : short2(partial_k, simdgroup_limit_m);
      const short2 b_limits = transpose_b
                                  ? short2(partial_k, simdgroup_limit_n)
                                  : short2(simdgroup_limit_n, partial_k);

      const int a_offset =
          transpose_a ? inner_k * leading_dimension_a : inner_k;
      const int b_offset =
          transpose_b ? inner_k : inner_k * leading_dimension_b;

      a_tile.load_checked(A + a_offset, leading_dimension_a, a_limits);
      b_tile.load_checked(B + b_offset, leading_dimension_b, b_limits);

      mxu_tiled_matmul(
          accumulator,
          a_tile,
          metal::bool_constant<transpose_a>{},
          b_tile,
          metal::bool_constant<transpose_b>{}
      );
    }
  }

  return accumulator;
}

} // namespace matmul
} // namespace uzu

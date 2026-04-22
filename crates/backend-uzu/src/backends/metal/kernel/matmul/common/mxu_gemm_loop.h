#pragma once

#include "fragment.h"
#include "mxu_fragment.h"
#include "../../generated/matmul.h"

using namespace metal;

namespace uzu {
namespace matmul {

template <
    typename T,
    ushort SIMDGROUP_BLOCK_M,
    ushort SIMDGROUP_BLOCK_N,
    ushort SIMDGROUP_BLOCK_K,
    ushort BLOCK_K,
    bool transpose_a,
    bool transpose_b,
    bool aligned_m,
    bool aligned_n,
    bool aligned_k,
    typename AccumulatorType = float>
METAL_FUNC auto gemm_loop(
    const device T* left_ptr,
    const device T* right_ptr,
    int leading_dimension_a,
    int leading_dimension_b,
    int K,
    int aligned_k_iterations,
    const short simdgroup_limit_m,
    const short simdgroup_limit_n,
    const thread ThreadContext& thread_context
) {
  constexpr ushort TILES_M = SIMDGROUP_BLOCK_M / MxuFragmentOps::FRAGMENT_ROWS;
  constexpr ushort TILES_N = SIMDGROUP_BLOCK_N / MxuFragmentOps::FRAGMENT_COLS;
  constexpr ushort TILES_K = SIMDGROUP_BLOCK_K / MxuFragmentOps::FRAGMENT_ROWS;

  constexpr ushort LEFT_TILE_ROWS = transpose_a ? TILES_K : TILES_M;
  constexpr ushort LEFT_TILE_COLS = transpose_a ? TILES_M : TILES_K;

  constexpr ushort RIGHT_TILE_ROWS = transpose_b ? TILES_N : TILES_K;
  constexpr ushort RIGHT_TILE_COLS = transpose_b ? TILES_K : TILES_N;

  Fragment<AccumulatorType, TILES_M, TILES_N, MxuFragmentOps> accumulator(
      thread_context
  );
  accumulator.clear();

  METAL_PRAGMA_NO_UNROLL
  for (int outer_k = 0; outer_k < aligned_k_iterations; outer_k++) {
    threadgroup_barrier(mem_flags::mem_none);

    METAL_PRAGMA_NO_UNROLL
    for (int inner_k = 0; inner_k < BLOCK_K; inner_k += SIMDGROUP_BLOCK_K) {
      Fragment<T, LEFT_TILE_ROWS, LEFT_TILE_COLS, MxuFragmentOps> left_tile(
          thread_context
      );
      Fragment<T, RIGHT_TILE_ROWS, RIGHT_TILE_COLS, MxuFragmentOps> right_tile(
          thread_context
      );

      const int left_offset =
          transpose_a ? inner_k * leading_dimension_a : inner_k;
      const int right_offset =
          transpose_b ? inner_k : inner_k * leading_dimension_b;

      if constexpr (aligned_m) {
        left_tile.load(left_ptr + left_offset, leading_dimension_a);
      } else {
        const short row_limit =
            transpose_a ? SIMDGROUP_BLOCK_K : simdgroup_limit_m;
        const short col_limit =
            transpose_a ? simdgroup_limit_m : SIMDGROUP_BLOCK_K;
        left_tile.load_safe(
            left_ptr + left_offset,
            leading_dimension_a,
            short2(col_limit, row_limit)
        );
      }

      if constexpr (aligned_n) {
        right_tile.load(right_ptr + right_offset, leading_dimension_b);
      } else {
        const short row_limit =
            transpose_b ? simdgroup_limit_n : SIMDGROUP_BLOCK_K;
        const short col_limit =
            transpose_b ? SIMDGROUP_BLOCK_K : simdgroup_limit_n;
        right_tile.load_safe(
            right_ptr + right_offset,
            leading_dimension_b,
            short2(col_limit, row_limit)
        );
      }

      MxuFragmentOps::template tile_matmul<transpose_a, transpose_b>(
          accumulator,
          left_tile,
          right_tile
      );
    }

    left_ptr += transpose_a ? (BLOCK_K * leading_dimension_a) : BLOCK_K;
    right_ptr += transpose_b ? BLOCK_K : (BLOCK_K * leading_dimension_b);
  }

  if constexpr (!aligned_k) {
    simdgroup_barrier(mem_flags::mem_none);

    const short remaining_k = K - aligned_k_iterations * BLOCK_K;

    METAL_PRAGMA_NO_UNROLL
    for (int inner_k = 0; inner_k < remaining_k; inner_k += SIMDGROUP_BLOCK_K) {
      Fragment<T, LEFT_TILE_ROWS, LEFT_TILE_COLS, MxuFragmentOps> left_tile(
          thread_context
      );
      Fragment<T, RIGHT_TILE_ROWS, RIGHT_TILE_COLS, MxuFragmentOps> right_tile(
          thread_context
      );

      const short safe_k = max(short(0), short(remaining_k - inner_k));

      const short2 left_limits = transpose_a
                                     ? short2(simdgroup_limit_m, safe_k)
                                     : short2(safe_k, simdgroup_limit_m);
      const short2 right_limits = transpose_b
                                      ? short2(safe_k, simdgroup_limit_n)
                                      : short2(simdgroup_limit_n, safe_k);

      const int left_offset =
          transpose_a ? inner_k * leading_dimension_a : inner_k;
      const int right_offset =
          transpose_b ? inner_k : inner_k * leading_dimension_b;

      left_tile
          .load_safe(left_ptr + left_offset, leading_dimension_a, left_limits);
      right_tile.load_safe(
          right_ptr + right_offset,
          leading_dimension_b,
          right_limits
      );

      MxuFragmentOps::template tile_matmul<transpose_a, transpose_b>(
          accumulator,
          left_tile,
          right_tile
      );
    }
  }

  return accumulator;
}

} // namespace matmul
} // namespace uzu

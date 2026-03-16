#pragma once

#include "loader.h"
#include "mma.h"
#include "../../../generated/matmul.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// Loop Alignment Helper
///////////////////////////////////////////////////////////////////////////////

template <bool M_aligned, bool N_aligned, bool K_aligned>
struct LoopAlignment {};

///////////////////////////////////////////////////////////////////////////////
// ThreadgroupGemm Kernel
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename U,
    int BLOCK_ROWS,
    int BLOCK_COLS,
    int BLOCK_DEPTH,
    int SIMDGROUPS_PER_ROW,
    int SIMDGROUPS_PER_COLUMN,
    bool transpose_a,
    bool transpose_b,
    bool MN_aligned,
    bool K_aligned,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct ThreadgroupGemm {
  METAL_CONST short THREADGROUP_PADDING_A = 16 / sizeof(T);
  METAL_CONST short THREADGROUP_PADDING_B = 16 / sizeof(T);
  METAL_CONST short THREADGROUP_MEMORY_SIZE_A =
      transpose_a ? BLOCK_DEPTH * (BLOCK_ROWS + THREADGROUP_PADDING_A) : BLOCK_ROWS * (BLOCK_DEPTH + THREADGROUP_PADDING_A);
  METAL_CONST short THREADGROUP_MEMORY_SIZE_B =
      transpose_b ? BLOCK_COLS * (BLOCK_DEPTH + THREADGROUP_PADDING_B) : BLOCK_DEPTH * (BLOCK_COLS + THREADGROUP_PADDING_B);
  METAL_CONST short THREADGROUP_MEMORY_SIZE = THREADGROUP_MEMORY_SIZE_A + THREADGROUP_MEMORY_SIZE_B;

  METAL_CONST short THREADGROUP_SIZE = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * 32;

  using LoaderAType = ThreadgroupLoader<
      T,
      transpose_a ? BLOCK_DEPTH : BLOCK_ROWS,
      transpose_a ? BLOCK_ROWS : BLOCK_DEPTH,
      transpose_a ? BLOCK_ROWS + THREADGROUP_PADDING_A : BLOCK_DEPTH + THREADGROUP_PADDING_A,
      !transpose_a,
      THREADGROUP_SIZE>;
  using LoaderBType = ThreadgroupLoader<
      T,
      transpose_b ? BLOCK_COLS : BLOCK_DEPTH,
      transpose_b ? BLOCK_DEPTH : BLOCK_COLS,
      transpose_b ? BLOCK_DEPTH + THREADGROUP_PADDING_B : BLOCK_COLS + THREADGROUP_PADDING_B,
      transpose_b,
      THREADGROUP_SIZE>;
  using ThreadgroupTileType = ThreadgroupTile<
      T,
      U,
      BLOCK_ROWS,
      BLOCK_COLS,
      BLOCK_DEPTH,
      SIMDGROUPS_PER_ROW,
      SIMDGROUPS_PER_COLUMN,
      transpose_a,
      transpose_b,
      transpose_a ? BLOCK_ROWS + THREADGROUP_PADDING_A : BLOCK_DEPTH + THREADGROUP_PADDING_A,
      transpose_b ? BLOCK_DEPTH + THREADGROUP_PADDING_B : BLOCK_COLS + THREADGROUP_PADDING_B,
      AccumType,
      Epilogue>;

  template <bool M_aligned, bool N_aligned, bool K_aligned_>
  static METAL_FUNC void gemm_loop(
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      const int gemm_k_iterations,
      thread LoaderAType& loader_a,
      thread LoaderBType& loader_b,
      thread ThreadgroupTileType& threadgroup_tile,
      thread const short& threadgroup_block_rows,
      thread const short& threadgroup_block_cols,
      thread const short& leftover_block_depth,
      LoopAlignment<M_aligned, N_aligned, K_aligned_> alignment = {}
  ) {
    (void)alignment;

    short2 tile_dimensions_a = transpose_a ? short2(threadgroup_block_rows, BLOCK_DEPTH) : short2(BLOCK_DEPTH, threadgroup_block_rows);
    short2 tile_dimensions_b = transpose_b ? short2(BLOCK_DEPTH, threadgroup_block_cols) : short2(threadgroup_block_cols, BLOCK_DEPTH);

    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (M_aligned) {
        loader_a.load_unsafe();
      } else {
        loader_a.load_safe(tile_dimensions_a);
      }

      if (N_aligned) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dimensions_b);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      threadgroup_tile.multiply_accumulate(a_shared, b_shared);

      loader_a.next();
      loader_b.next();
    }

    if (!K_aligned_) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      short2 last_tile_dimensions_a =
          transpose_a ? short2(threadgroup_block_rows, leftover_block_depth) : short2(leftover_block_depth, threadgroup_block_rows);
      short2 last_tile_dimensions_b =
          transpose_b ? short2(leftover_block_depth, threadgroup_block_cols) : short2(threadgroup_block_cols, leftover_block_depth);

      loader_a.load_safe(last_tile_dimensions_a);
      loader_b.load_safe(last_tile_dimensions_b);

      threadgroup_barrier(mem_flags::mem_threadgroup);

      threadgroup_tile.multiply_accumulate(a_shared, b_shared);
    }
  }

  static METAL_FUNC void run(
      const device T* a,
      const device T* b,
      device U* d,
      const constant GemmParams* params,
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      uint simd_lane_id [[thread_index_in_simdgroup]],
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint2 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]]
  ) {
    (void)lid;

    const int swizzled_row_id = ((tid.y) << params->swizzle_log) +
                      ((tid.x) & ((1 << params->swizzle_log) - 1));
    const int swizzled_col_id = (tid.x) >> params->swizzle_log;

    if (params->threadgroups_per_row <= swizzled_col_id || params->threadgroups_per_column <= swizzled_row_id) {
      return;
    }

    threadgroup_barrier(mem_flags::mem_none);

    const int output_row = swizzled_row_id * BLOCK_ROWS;
    const int output_col = swizzled_col_id * BLOCK_COLS;
    const size_t output_row_long = size_t(output_row);
    const size_t output_col_long = size_t(output_col);

    a += transpose_a ? output_row_long : output_row_long * params->leading_dimension_a;
    b += transpose_b ? output_col_long * params->leading_dimension_b : output_col_long;
    d += output_row_long * params->leading_dimension_d + output_col_long;

    thread LoaderAType loader_a(a, params->leading_dimension_a, a_shared, simd_group_id, simd_lane_id);
    thread LoaderBType loader_b(b, params->leading_dimension_b, b_shared, simd_group_id, simd_lane_id);

    thread ThreadgroupTileType threadgroup_tile(simd_group_id, simd_lane_id);

    int gemm_k_iterations = params->aligned_inner_iterations;

    ///////////////////////////////////////////////////////////////////////////
    // MNK aligned loop
    if (MN_aligned) {
      for (int k = 0; k < gemm_k_iterations; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup_tile.multiply_accumulate(a_shared, b_shared);

        loader_a.next();
        loader_b.next();
      }

      threadgroup_barrier(mem_flags::mem_none);

      if (!K_aligned) {
        int leftover_block_depth = params->K - params->aligned_inner_iterations * BLOCK_DEPTH;
        short2 tile_dimensions_a = transpose_a ? short2(BLOCK_ROWS, leftover_block_depth) : short2(leftover_block_depth, BLOCK_ROWS);
        short2 tile_dimensions_b = transpose_b ? short2(leftover_block_depth, BLOCK_COLS) : short2(BLOCK_COLS, leftover_block_depth);

        loader_a.load_safe(tile_dimensions_a);
        loader_b.load_safe(tile_dimensions_b);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup_tile.multiply_accumulate(a_shared, b_shared);
      }

      threadgroup_tile.store_result(d, params->leading_dimension_d);
      return;
    }
    ///////////////////////////////////////////////////////////////////////////
    // MN unaligned loop
    else {
      short threadgroup_block_rows = min(BLOCK_ROWS, params->M - output_row);
      short threadgroup_block_cols = min(BLOCK_COLS, params->N - output_col);
      short leftover_block_depth = params->K - params->aligned_inner_iterations * BLOCK_DEPTH;

      if (threadgroup_block_rows == BLOCK_ROWS && threadgroup_block_cols == BLOCK_COLS) {
        gemm_loop<true, true, K_aligned>(
            a_shared,
            b_shared,
            gemm_k_iterations,
            loader_a,
            loader_b,
            threadgroup_tile,
            threadgroup_block_rows,
            threadgroup_block_cols,
            leftover_block_depth
        );

        threadgroup_tile.store_result(d, params->leading_dimension_d);
        return;

      } else if (threadgroup_block_cols == BLOCK_COLS) {
        gemm_loop<false, true, K_aligned>(
            a_shared,
            b_shared,
            gemm_k_iterations,
            loader_a,
            loader_b,
            threadgroup_tile,
            threadgroup_block_rows,
            threadgroup_block_cols,
            leftover_block_depth
        );

        threadgroup_tile.store_result_safe(d, params->leading_dimension_d, short2(threadgroup_block_cols, threadgroup_block_rows));
        return;

      } else if (threadgroup_block_rows == BLOCK_ROWS) {
        gemm_loop<true, false, K_aligned>(
            a_shared,
            b_shared,
            gemm_k_iterations,
            loader_a,
            loader_b,
            threadgroup_tile,
            threadgroup_block_rows,
            threadgroup_block_cols,
            leftover_block_depth
        );

        threadgroup_tile.store_result_safe(d, params->leading_dimension_d, short2(threadgroup_block_cols, threadgroup_block_rows));
        return;

      } else {
        gemm_loop<false, false, K_aligned>(
            a_shared,
            b_shared,
            gemm_k_iterations,
            loader_a,
            loader_b,
            threadgroup_tile,
            threadgroup_block_rows,
            threadgroup_block_cols,
            leftover_block_depth
        );

        threadgroup_tile.store_result_safe(d, params->leading_dimension_d, short2(threadgroup_block_cols, threadgroup_block_rows));
        return;
      }
    }
  }
};

} // namespace matmul
} // namespace uzu

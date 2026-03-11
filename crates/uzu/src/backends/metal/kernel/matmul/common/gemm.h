#pragma once

#include "loader.h"
#include "mma.h"
#include "mpp_gemm_utilities.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// Loop Alignment Helper
///////////////////////////////////////////////////////////////////////////////

template <bool M_aligned, bool N_aligned, bool K_aligned>
struct LoopAlignment {};

///////////////////////////////////////////////////////////////////////////////
// GEMM Kernel
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename U,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int WARPS_M,
    int WARPS_N,
    bool transpose_a,
    bool transpose_b,
    bool MN_aligned,
    bool K_aligned,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct GEMMKernel {
  MTL_CONST short THREADGROUP_PADDING_A = 16 / sizeof(T);
  MTL_CONST short THREADGROUP_PADDING_B = 16 / sizeof(T);
  MTL_CONST short THREADGROUP_MEMORY_SIZE_A =
      transpose_a ? BLOCK_K * (BLOCK_M + THREADGROUP_PADDING_A) : BLOCK_M * (BLOCK_K + THREADGROUP_PADDING_A);
  MTL_CONST short THREADGROUP_MEMORY_SIZE_B =
      transpose_b ? BLOCK_N * (BLOCK_K + THREADGROUP_PADDING_B) : BLOCK_K * (BLOCK_N + THREADGROUP_PADDING_B);
  MTL_CONST short THREADGROUP_MEMORY_SIZE = THREADGROUP_MEMORY_SIZE_A + THREADGROUP_MEMORY_SIZE_B;

  MTL_CONST short THREADGROUP_SIZE = WARPS_M * WARPS_N * 32;

  using LoaderAType = BlockLoader<
      T,
      transpose_a ? BLOCK_K : BLOCK_M,
      transpose_a ? BLOCK_M : BLOCK_K,
      transpose_a ? BLOCK_M + THREADGROUP_PADDING_A : BLOCK_K + THREADGROUP_PADDING_A,
      !transpose_a,
      THREADGROUP_SIZE>;
  using LoaderBType = BlockLoader<
      T,
      transpose_b ? BLOCK_N : BLOCK_K,
      transpose_b ? BLOCK_K : BLOCK_N,
      transpose_b ? BLOCK_K + THREADGROUP_PADDING_B : BLOCK_N + THREADGROUP_PADDING_B,
      transpose_b,
      THREADGROUP_SIZE>;
  using MMAType = BlockMMA<
      T,
      U,
      BLOCK_M,
      BLOCK_N,
      BLOCK_K,
      WARPS_M,
      WARPS_N,
      transpose_a,
      transpose_b,
      transpose_a ? BLOCK_M + THREADGROUP_PADDING_A : BLOCK_K + THREADGROUP_PADDING_A,
      transpose_b ? BLOCK_K + THREADGROUP_PADDING_B : BLOCK_N + THREADGROUP_PADDING_B,
      AccumType,
      Epilogue>;

  /* Main kernel function */
  template <bool M_aligned, bool N_aligned, bool K_aligned_>
  static METAL_FUNC void gemm_loop(
      threadgroup T* left_shared,
      threadgroup T* right_shared,
      const int gemm_k_iterations,
      thread LoaderAType& loader_a,
      thread LoaderBType& loader_b,
      thread MMAType& mma_operation,
      thread const short& threadgroup_block_m,
      thread const short& threadgroup_block_n,
      thread const short& leftover_block_k,
      LoopAlignment<M_aligned, N_aligned, K_aligned_> l = {}
  ) {
    // Appease the compiler
    (void)l;

    short2 tile_dims_A = transpose_a ? short2(threadgroup_block_m, BLOCK_K) : short2(BLOCK_K, threadgroup_block_m);
    short2 tile_dims_B = transpose_b ? short2(BLOCK_K, threadgroup_block_n) : short2(threadgroup_block_n, BLOCK_K);

    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Load elements into threadgroup
      if (M_aligned) {
        loader_a.load_unchecked();
      } else {
        loader_a.load_checked(tile_dims_A);
      }

      if (N_aligned) {
        loader_b.load_unchecked();
      } else {
        loader_b.load_checked(tile_dims_B);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Multiply and accumulate threadgroup elements
      mma_operation.mma(left_shared, right_shared);

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }

    if (!K_aligned_) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      short2 tile_dims_A_last =
          transpose_a ? short2(threadgroup_block_m, leftover_block_k) : short2(leftover_block_k, threadgroup_block_m);
      short2 tile_dims_B_last =
          transpose_b ? short2(leftover_block_k, threadgroup_block_n) : short2(threadgroup_block_n, leftover_block_k);

      loader_a.load_checked(tile_dims_A_last);
      loader_b.load_checked(tile_dims_B_last);

      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_operation.mma(left_shared, right_shared);
    }
  }

  /* Main kernel function */
  static METAL_FUNC void run(
      const device T* left_matrix,
      const device T* right_matrix,
      device U* output_matrix,
      const constant GEMMParams* params,
      threadgroup T* left_shared,
      threadgroup T* right_shared,
      uint simd_lane_id [[thread_index_in_simdgroup]],
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint3 threadgroup_position [[threadgroup_position_in_grid]],
      uint3 thread_position [[thread_position_in_threadgroup]]
  ) {
    // Pacifying compiler
    (void)thread_position;

    const int swizzle_size = 1 << params->swizzle_log;
    const int tid_y = threadgroup_position.y * swizzle_size + (threadgroup_position.x % swizzle_size);
    const int tid_x = threadgroup_position.x / swizzle_size;

    if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
      return;
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Find block in left_matrix, right_matrix, output_matrix
    const int block_row_start = tid_y * BLOCK_M;
    const int block_col_start = tid_x * BLOCK_N;
    const size_t block_row_start_long = size_t(block_row_start);
    const size_t block_col_start_long = size_t(block_col_start);

    left_matrix += transpose_a ? block_row_start_long : block_row_start_long * params->leading_dim_a;
    right_matrix += transpose_b ? block_col_start_long * params->leading_dim_b : block_col_start_long;
    output_matrix += block_row_start_long * params->leading_dim_d + block_col_start_long;

    // Prepare threadgroup loading operations
    thread LoaderAType loader_a(left_matrix, params->leading_dim_a, left_shared, simd_group_id, simd_lane_id);
    thread LoaderBType loader_b(right_matrix, params->leading_dim_b, right_shared, simd_group_id, simd_lane_id);

    // Prepare threadgroup mma operation
    thread MMAType mma_operation(simd_group_id, simd_lane_id);

    int gemm_k_iterations = params->gemm_k_iterations_aligned;

    ///////////////////////////////////////////////////////////////////////////
    // MNK aligned loop
    if (MN_aligned) {
      for (int k = 0; k < gemm_k_iterations; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Load elements into threadgroup
        loader_a.load_unchecked();
        loader_b.load_unchecked();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_operation.mma(left_shared, right_shared);

        // Prepare for next iteration
        loader_a.next();
        loader_b.next();
      }

      threadgroup_barrier(mem_flags::mem_none);

      // Loop tail
      if (!K_aligned) {
        int leftover_block_k = params->K - params->gemm_k_iterations_aligned * BLOCK_K;
        short2 tile_dims_A = transpose_a ? short2(BLOCK_M, leftover_block_k) : short2(leftover_block_k, BLOCK_M);
        short2 tile_dims_B = transpose_b ? short2(leftover_block_k, BLOCK_N) : short2(BLOCK_N, leftover_block_k);

        loader_a.load_checked(tile_dims_A);
        loader_b.load_checked(tile_dims_B);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_operation.mma(left_shared, right_shared);
      }

      // Store results to device memory
      mma_operation.store_result(output_matrix, params->leading_dim_d);
      return;
    }
    ///////////////////////////////////////////////////////////////////////////
    // MN unaligned loop
    else { // Loop over K - unaligned case
      short threadgroup_block_m = min(BLOCK_M, params->M - block_row_start);
      short threadgroup_block_n = min(BLOCK_N, params->N - block_col_start);
      short leftover_block_k =
          params->K - params->gemm_k_iterations_aligned * BLOCK_K;

      if (threadgroup_block_m == BLOCK_M && threadgroup_block_n == BLOCK_N) {
        gemm_loop<true, true, K_aligned>(
            left_shared,
            right_shared,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_operation,
            threadgroup_block_m,
            threadgroup_block_n,
            leftover_block_k
        );

        mma_operation.store_result(output_matrix, params->leading_dim_d);
        return;

      } else if (threadgroup_block_n == BLOCK_N) {
        gemm_loop<false, true, K_aligned>(
            left_shared,
            right_shared,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_operation,
            threadgroup_block_m,
            threadgroup_block_n,
            leftover_block_k
        );

        mma_operation.store_result_checked(output_matrix, params->leading_dim_d, short2(threadgroup_block_n, threadgroup_block_m));
        return;

      } else if (threadgroup_block_m == BLOCK_M) {
        gemm_loop<true, false, K_aligned>(
            left_shared,
            right_shared,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_operation,
            threadgroup_block_m,
            threadgroup_block_n,
            leftover_block_k
        );

        mma_operation.store_result_checked(output_matrix, params->leading_dim_d, short2(threadgroup_block_n, threadgroup_block_m));
        return;

      } else {
        gemm_loop<false, false, K_aligned>(
            left_shared,
            right_shared,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_operation,
            threadgroup_block_m,
            threadgroup_block_n,
            leftover_block_k
        );

        mma_operation.store_result_checked(output_matrix, params->leading_dim_d, short2(threadgroup_block_n, threadgroup_block_m));
        return;
      }
    }
  }
};

} // namespace matmul
} // namespace uzu

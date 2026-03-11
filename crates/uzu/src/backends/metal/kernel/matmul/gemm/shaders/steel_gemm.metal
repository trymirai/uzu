// clang-format off
#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include "../../common/steel/gemm/gemm.h"

using namespace steel;

// Upper bounds for threadgroup memory (in elements of T).
// Max across all tile/type combos: 64x32x32 with half (padding=8).
// threadgroup_mem_size_a = BLOCK_M * (BLOCK_K + padding) = 64 * (32 + 8) = 2560
// threadgroup_mem_size_b = BLOCK_N * (BLOCK_K + padding) = 32 * (32 + 8) = 1280
// For 64x64x16 with half: a=64*(16+8)=1536, b=64*(16+8)=1536
// Max a = 2560, max b = 1536
#define GEMM_MAX_TGP_A 2560
#define GEMM_MAX_TGP_B 1536

namespace uzu {
namespace matmul {
using GEMMParams = steel::GEMMParams;
} // namespace matmul
} // namespace uzu

///////////////////////////////////////////////////////////////////////////////
// GEMM implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename AccumType = float>
METAL_FUNC void gemm_impl(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant GEMMParams* params,
    const int BLOCK_M,
    const int BLOCK_N,
    const int BLOCK_K,
    const int WARPS_M,
    const int WARPS_N,
    const bool align_m,
    const bool align_n,
    const bool align_k,
    threadgroup T* left_shared,
    threadgroup T* right_shared,
    uint simd_lane_id,
    uint simd_group_id,
    uint3 threadgroup_position,
    uint3 thread_position
) {
  (void)thread_position;

  // Hardcoded: transpose_a = false, transpose_b = true
  const bool transpose_a = false;
  const bool transpose_b = true;

  const short threadgroup_padding = 16 / sizeof(T);
  const short threadgroup_leading_dim_a = BLOCK_K + threadgroup_padding;  // transpose_a=false: BLOCK_K + pad
  const short threadgroup_leading_dim_b = BLOCK_K + threadgroup_padding;  // transpose_b=true:  BLOCK_K + pad
  const short threadgroup_size = WARPS_M * WARPS_N * 32;

  // Find block
  const int swizzle_size = 1 << params->swizzle_log;
  const int tid_y = ((threadgroup_position.y) * swizzle_size) +
                    ((threadgroup_position.x) % swizzle_size);
  const int tid_x = (threadgroup_position.x) / swizzle_size;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  // Batch offset (non-batched path only)
  left_matrix += params->batch_stride_a * threadgroup_position.z;
  right_matrix += params->batch_stride_b * threadgroup_position.z;
  output_matrix += params->batch_stride_d * threadgroup_position.z;

  threadgroup_barrier(mem_flags::mem_none);

  // Find block in left_matrix, right_matrix, output_matrix
  const int block_row_start = tid_y * BLOCK_M;
  const int block_col_start = tid_x * BLOCK_N;
  const size_t block_row_start_long = size_t(block_row_start);
  const size_t block_col_start_long = size_t(block_col_start);

  // transpose_a=false: left_matrix += block_row_start * leading_dim_a
  left_matrix += block_row_start_long * params->leading_dim_a;
  // transpose_b=true: right_matrix += block_col_start * leading_dim_b
  right_matrix += block_col_start_long * params->leading_dim_b;
  output_matrix += block_row_start_long * params->leading_dim_d + block_col_start_long;

  // Construct loader and MMA objects with runtime tile params
  // transpose_a=false: loader_a BROWS=BLOCK_M, BCOLS=BLOCK_K, dst_ld=BLOCK_K+pad, reduction_dim=1
  thread BlockLoader<T> loader_a(
      left_matrix, params->leading_dim_a, left_shared,
      simd_group_id, simd_lane_id,
      BLOCK_M, BLOCK_K, threadgroup_leading_dim_a, 1, threadgroup_size);

  // transpose_b=true: loader_b BROWS=BLOCK_N, BCOLS=BLOCK_K, dst_ld=BLOCK_K+pad, reduction_dim=1
  thread BlockLoader<T> loader_b(
      right_matrix, params->leading_dim_b, right_shared,
      simd_group_id, simd_lane_id,
      BLOCK_N, BLOCK_K, threadgroup_leading_dim_b, 1, threadgroup_size);

  thread BlockMMA<T, T, AccumType> mma_operation(
      simd_group_id, simd_lane_id,
      BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N,
      transpose_a, transpose_b,
      threadgroup_leading_dim_a, threadgroup_leading_dim_b);

  const short threadgroup_block_m = align_m ? BLOCK_M : short(min(BLOCK_M, params->M - block_row_start));
  const short threadgroup_block_n = align_n ? BLOCK_N : short(min(BLOCK_N, params->N - block_col_start));

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  // Do unaligned K iterations first
  if (!align_k) {
    const int k_last = params->gemm_k_iterations_aligned * BLOCK_K;
    const int k_remain = params->K - k_last;
    // transpose_a=false: k_jump_a = k_last
    const size_t k_jump_a = size_t(k_last);
    // transpose_b=true: k_jump_b = k_last
    const size_t k_jump_b = size_t(k_last);

    loader_a.source += k_jump_a;
    loader_b.source += k_jump_b;

    // transpose_a=false: tile_dims_a = (k_remain, threadgroup_block_m)
    const short2 tile_dims_a = short2(k_remain, threadgroup_block_m);
    // transpose_b=true: tile_dims_b = (k_remain, threadgroup_block_n)
    const short2 tile_dims_b = short2(k_remain, threadgroup_block_n);

    loader_a.load_checked(tile_dims_a);
    loader_b.load_checked(tile_dims_b);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_operation.mma(left_shared, right_shared);

    loader_a.source -= k_jump_a;
    loader_b.source -= k_jump_b;
  }

  // MNK aligned loop
  if (align_m && align_n) {
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unchecked();
      loader_b.load_unchecked();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_operation.mma(left_shared, right_shared);
      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);
    return mma_operation.store_result(output_matrix, params->leading_dim_d);
  } else {
    const short leftover_block_k = 0;

    if ((align_m || threadgroup_block_m == BLOCK_M) && (align_n || threadgroup_block_n == BLOCK_N)) {
      gemm_loop<T, T, AccumType>(
          left_shared, right_shared, gemm_k_iterations, loader_a, loader_b,
          mma_operation, threadgroup_block_m, threadgroup_block_n, leftover_block_k,
          BLOCK_K, transpose_a, transpose_b, true, true, true);
      return mma_operation.store_result(output_matrix, params->leading_dim_d);
    } else if (align_n || threadgroup_block_n == BLOCK_N) {
      gemm_loop<T, T, AccumType>(
          left_shared, right_shared, gemm_k_iterations, loader_a, loader_b,
          mma_operation, threadgroup_block_m, threadgroup_block_n, leftover_block_k,
          BLOCK_K, transpose_a, transpose_b, false, true, true);
      return mma_operation.store_result_checked(output_matrix, params->leading_dim_d, short2(threadgroup_block_n, threadgroup_block_m));
    } else if (align_m || threadgroup_block_m == BLOCK_M) {
      gemm_loop<T, T, AccumType>(
          left_shared, right_shared, gemm_k_iterations, loader_a, loader_b,
          mma_operation, threadgroup_block_m, threadgroup_block_n, leftover_block_k,
          BLOCK_K, transpose_a, transpose_b, true, false, true);
      return mma_operation.store_result_checked(output_matrix, params->leading_dim_d, short2(threadgroup_block_n, threadgroup_block_m));
    } else {
      gemm_loop<T, T, AccumType>(
          left_shared, right_shared, gemm_k_iterations, loader_a, loader_b,
          mma_operation, threadgroup_block_m, threadgroup_block_n, leftover_block_k,
          BLOCK_K, transpose_a, transpose_b, false, false, true);
      return mma_operation.store_result_checked(output_matrix, params->leading_dim_d, short2(threadgroup_block_n, threadgroup_block_m));
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Unified DSL kernel
///////////////////////////////////////////////////////////////////////////////

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemm)(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup T left_shared[GEMM_MAX_TGP_A],
    threadgroup T right_shared[GEMM_MAX_TGP_B],
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint block_depth SPECIALIZE,
    const uint warps_per_row SPECIALIZE,
    const uint warps_per_col SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    const Simd simd
) {
  gemm_impl<T, float>(
      left_matrix, right_matrix, output_matrix, params,
      block_rows, block_cols, block_depth,
      warps_per_row, warps_per_col,
      align_m, align_n, align_k,
      left_shared, right_shared,
      simd.lane_idx, simd.group_idx,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, thread_z)
  );
}

// clang-format on

// clang-format off
#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include "../../common/gemm.h"

using namespace uzu::matmul;

// Upper bounds for threadgroup memory (in elements of T).
// Max across all tile/type combos: 64x32x32 with half (padding=8).
// threadgroup_mem_size_a = BLOCK_M * (BLOCK_K + padding) = 64 * (32 + 8) = 2560
// threadgroup_mem_size_b = BLOCK_N * (BLOCK_K + padding) = 32 * (32 + 8) = 1280
// For 64x64x16 with half: a=64*(16+8)=1536, b=64*(16+8)=1536
// Max a = 2560, max b = 1536
#define GEMM_MAX_TGP_A 2560
#define GEMM_MAX_TGP_B 1536

///////////////////////////////////////////////////////////////////////////////
// Dispatch helper: calls GEMMKernel::run() with the right template params.
// transpose_a = false, transpose_b = true (hardcoded).
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int WARPS_M,
    int WARPS_N,
    bool MN_aligned,
    bool K_aligned>
METAL_FUNC void gemm_dispatch(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant GEMMParams* params,
    threadgroup T* left_shared,
    threadgroup T* right_shared,
    uint simd_lane_id,
    uint simd_group_id,
    uint3 threadgroup_position,
    uint3 thread_position
) {
  // Apply batch offsets before calling GEMMKernel::run()
  left_matrix += params->batch_stride_a * threadgroup_position.z;
  right_matrix += params->batch_stride_b * threadgroup_position.z;
  output_matrix += params->batch_stride_d * threadgroup_position.z;

  GEMMKernel<T, T, BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N,
             /*transpose_a=*/false, /*transpose_b=*/true,
             MN_aligned, K_aligned>::run(
      left_matrix, right_matrix, output_matrix, params,
      left_shared, right_shared,
      simd_lane_id, simd_group_id,
      threadgroup_position, thread_position);
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
    const constant GEMMParams* params,
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
  const bool mn_aligned = align_m && align_n;
  const uint3 tg_pos = uint3(group_x, group_y, group_z);
  const uint3 th_pos = uint3(thread_x, thread_y, thread_z);

  // Dispatch to the correct GEMMKernel instantiation based on tile config.
  // Tile configs from specialization.rs: (64,64,16,2,2), (64,32,32,2,2), (32,64,16,2,2)
  if (block_rows == 64 && block_cols == 64 && block_depth == 16 &&
      warps_per_row == 2 && warps_per_col == 2) {
    if (mn_aligned && align_k) {
      gemm_dispatch<T, 64, 64, 16, 2, 2, true, true>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    } else if (mn_aligned) {
      gemm_dispatch<T, 64, 64, 16, 2, 2, true, false>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    } else if (align_k) {
      gemm_dispatch<T, 64, 64, 16, 2, 2, false, true>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    } else {
      gemm_dispatch<T, 64, 64, 16, 2, 2, false, false>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    }
  } else if (block_rows == 64 && block_cols == 32 && block_depth == 32 &&
             warps_per_row == 2 && warps_per_col == 2) {
    if (mn_aligned && align_k) {
      gemm_dispatch<T, 64, 32, 32, 2, 2, true, true>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    } else if (mn_aligned) {
      gemm_dispatch<T, 64, 32, 32, 2, 2, true, false>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    } else if (align_k) {
      gemm_dispatch<T, 64, 32, 32, 2, 2, false, true>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    } else {
      gemm_dispatch<T, 64, 32, 32, 2, 2, false, false>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    }
  } else if (block_rows == 32 && block_cols == 64 && block_depth == 16 &&
             warps_per_row == 2 && warps_per_col == 2) {
    if (mn_aligned && align_k) {
      gemm_dispatch<T, 32, 64, 16, 2, 2, true, true>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    } else if (mn_aligned) {
      gemm_dispatch<T, 32, 64, 16, 2, 2, true, false>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    } else if (align_k) {
      gemm_dispatch<T, 32, 64, 16, 2, 2, false, true>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    } else {
      gemm_dispatch<T, 32, 64, 16, 2, 2, false, false>(
          left_matrix, right_matrix, output_matrix, params,
          left_shared, right_shared, simd.lane_idx, simd.group_idx, tg_pos, th_pos);
    }
  }
}

// clang-format on

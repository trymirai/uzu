// clang-format off
#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include "../../common/params.h"
#include "../../common/cooperative_tensor_gemm.h"
#include "../../common/loader.h"
#include "../../common/gemm_mpp_core.h"

using namespace uzu::matmul;

///////////////////////////////////////////////////////////////////////////////
// Threadgroup staging constants (T is #define'd by the DSL wrapper)
///////////////////////////////////////////////////////////////////////////////

#define PREFETCH_K_SIZE (((26624 / (128 * 2 * (int)sizeof(T))) / 16) * 16)
#define THREADGROUP_PADDING ((short)(16 / sizeof(T)))
#define THREADGROUP_LEADING_DIMENSION ((short)(PREFETCH_K_SIZE + THREADGROUP_PADDING))
#define THREADGROUP_TILE_SIZE (128 * THREADGROUP_LEADING_DIMENSION)

///////////////////////////////////////////////////////////////////////////////
// MPP GEMM implementation -- templated over tile config
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    short BLOCK_M,
    short BLOCK_N,
    short BLOCK_K,
    short WARPS_M,
    short WARPS_N,
    short SUBTILE_ROWS,
    short SUBTILE_COLS,
    short MATMUL_K_STEP>
METAL_FUNC void gemm_mpp_impl(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant GEMMParams* params,
    const bool align_m,
    const bool align_n,
    const bool align_k,
    threadgroup T* left_shared,
    threadgroup T* right_shared,
    uint simd_group_id,
    uint simd_lane_id,
    uint3 threadgroup_position
) {
  using AccumulatorType = float;

  constexpr short SIMDGROUP_M = BLOCK_M / WARPS_M;
  constexpr short SIMDGROUP_N = BLOCK_N / WARPS_N;
  constexpr short TILES_M = SIMDGROUP_M / SUBTILE_ROWS;
  constexpr short TILES_N = SIMDGROUP_N / SUBTILE_COLS;
  constexpr short THREADGROUP_SIZE = WARPS_M * WARPS_N * SIMD_SIZE;

  constexpr short PREFETCH_K = short(PREFETCH_K_SIZE);
  constexpr short THREADGROUP_LD = PREFETCH_K + short(16 / sizeof(T));

  const int swizzle_size = 1 << params->swizzle_log;
  const int tid_y = ((threadgroup_position.y) * swizzle_size) +
                    ((threadgroup_position.x) % swizzle_size);
  const int tid_x = (threadgroup_position.x) / swizzle_size;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  left_matrix += params->batch_stride_a * threadgroup_position.z;
  right_matrix += params->batch_stride_b * threadgroup_position.z;
  output_matrix += params->batch_stride_d * threadgroup_position.z;

  threadgroup_barrier(mem_flags::mem_none);

  const int block_row_start = tid_y * BLOCK_M;
  const int block_col_start = tid_x * BLOCK_N;
  const size_t block_row_start_long = size_t(block_row_start);
  const size_t block_col_start_long = size_t(block_col_start);

  const device T* left_block_ptr = left_matrix + block_row_start_long * params->leading_dim_a;
  const device T* right_block_ptr = right_matrix + block_col_start_long * params->leading_dim_b;

  // --- Fallback: direct device reads when staging gives no benefit ---
  if constexpr (PREFETCH_K <= MATMUL_K_STEP) {
    const short tile_row_offset = SIMDGROUP_M * (simd_group_id / WARPS_N);
    const short tile_col_offset = SIMDGROUP_N * (simd_group_id % WARPS_N);

    device T* output_ptr = output_matrix + block_row_start_long * params->leading_dim_d + block_col_start_long + tile_row_offset * params->leading_dim_d + tile_col_offset;

    const int simdgroup_limit_m_int =
        align_m ? int(SIMDGROUP_M) : min(int(SIMDGROUP_M), params->M - (block_row_start + tile_row_offset));
    const short simdgroup_limit_m = short(simdgroup_limit_m_int);

    const int simdgroup_limit_n_int =
        align_n ? int(SIMDGROUP_N) : min(int(SIMDGROUP_N), params->N - (block_col_start + tile_col_offset));
    const short simdgroup_limit_n = short(simdgroup_limit_n_int);

    const bool is_unaligned_m = align_m ? false : (simdgroup_limit_m != SIMDGROUP_M);
    const bool is_unaligned_n = align_n ? false : (simdgroup_limit_n != SIMDGROUP_N);

    const device T* left_simdgroup_ptr = left_block_ptr + tile_row_offset * params->leading_dim_a;
    const device T* right_simdgroup_ptr = right_block_ptr + tile_col_offset * params->leading_dim_b;

    PRAGMA_UNROLL
    for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
      PRAGMA_UNROLL
      for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
        const short row_offset = tile_m * SUBTILE_ROWS;
        const short col_offset = tile_n * SUBTILE_COLS;

        const short m_limit = is_unaligned_m ? short(max(0, int(simdgroup_limit_m) - row_offset)) : SUBTILE_ROWS;
        const short n_limit = is_unaligned_n ? short(max(0, int(simdgroup_limit_n) - col_offset)) : SUBTILE_COLS;
        if (m_limit <= 0 || n_limit <= 0) continue;

        const device T* left_subtile_ptr = left_simdgroup_ptr + row_offset * params->leading_dim_a;
        const device T* right_subtile_ptr = right_simdgroup_ptr + col_offset * params->leading_dim_b;
        device T* output_subtile_ptr = output_ptr + row_offset * params->leading_dim_d + col_offset;

        const bool subtile_aligned_m = !is_unaligned_m || (m_limit == SUBTILE_ROWS);
        const bool subtile_aligned_n = !is_unaligned_n || (n_limit == SUBTILE_COLS);

        if (subtile_aligned_m && subtile_aligned_n && align_k) {
          cooperative_tensor_gemm<SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP, AccumulatorType, T, T, T,
                                 false, true, true, true, true>(
              left_subtile_ptr, params->leading_dim_a, right_subtile_ptr, params->leading_dim_b, output_subtile_ptr, params->leading_dim_d,
              params->K, m_limit, n_limit);
        } else if (subtile_aligned_m && subtile_aligned_n) {
          cooperative_tensor_gemm<SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP, AccumulatorType, T, T, T,
                                 false, true, true, true, false>(
              left_subtile_ptr, params->leading_dim_a, right_subtile_ptr, params->leading_dim_b, output_subtile_ptr, params->leading_dim_d,
              params->K, m_limit, n_limit);
        } else {
          cooperative_tensor_gemm<SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP, AccumulatorType, T, T, T,
                                 false, true, false, false, false>(
              left_subtile_ptr, params->leading_dim_a, right_subtile_ptr, params->leading_dim_b, output_subtile_ptr, params->leading_dim_d,
              params->K, m_limit, n_limit);
        }
      }
    }
    return;
  }

  // --- Staged path: construct loaders and delegate to gemm_mpp_staged ---

  BlockLoader<T, BLOCK_M, PREFETCH_K, THREADGROUP_LD, 1, THREADGROUP_SIZE> loader_a(
      left_block_ptr, params->leading_dim_a, left_shared, ushort(simd_group_id), ushort(simd_lane_id));
  BlockLoader<T, BLOCK_N, PREFETCH_K, THREADGROUP_LD, 1, THREADGROUP_SIZE> loader_b(
      right_block_ptr, params->leading_dim_b, right_shared, ushort(simd_group_id), ushort(simd_lane_id));

  gemm_mpp_staged<T, decltype(loader_a), decltype(loader_b),
                  BLOCK_M, BLOCK_N, PREFETCH_K, THREADGROUP_LD, WARPS_M, WARPS_N,
                  SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP>(
      loader_a, loader_b,
      output_matrix, params,
      align_m, align_n,
      block_row_start, block_col_start,
      left_shared, right_shared,
      simd_group_id, simd_lane_id);
}

///////////////////////////////////////////////////////////////////////////////
// DSL kernel entry point
///////////////////////////////////////////////////////////////////////////////

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemmMpp)(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup T left_shared[THREADGROUP_TILE_SIZE],
    threadgroup T right_shared[THREADGROUP_TILE_SIZE],
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint block_depth SPECIALIZE,
    const uint warps_per_row SPECIALIZE,
    const uint warps_per_col SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const bool use_native_fragment_layout SPECIALIZE,
    const uint subtile_rows SPECIALIZE,
    const uint subtile_cols SPECIALIZE,
    const uint matmul_k_step SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(4),
    const uint thread_z THREADS(4),
    const Simd simd
) {
  if (simd.group_idx >= warps_per_row * warps_per_col) {
    return;
  }

  if (subtile_rows == 16 && subtile_cols == 32 && matmul_k_step == 16) {
    if (block_rows == 128 && block_cols == 128 && block_depth == 512 &&
        warps_per_row == 4 && warps_per_col == 4) {
      gemm_mpp_impl<T, 128, 128, 512, 4, 4, 16, 32, 16>(
          left_matrix, right_matrix, output_matrix, params,
          align_m, align_n, align_k,
          left_shared, right_shared,
          simd.group_idx, simd.lane_idx,
          uint3(group_x, group_y, group_z));
    } else {
      gemm_mpp_impl<T, 64, 64, 256, 2, 2, 16, 32, 16>(
          left_matrix, right_matrix, output_matrix, params,
          align_m, align_n, align_k,
          left_shared, right_shared,
          simd.group_idx, simd.lane_idx,
          uint3(group_x, group_y, group_z));
    }
  }
}

// clang-format on

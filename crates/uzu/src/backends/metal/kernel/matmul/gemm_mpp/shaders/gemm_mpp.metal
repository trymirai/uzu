// clang-format off
#include "../../../common/utils.h"
#include "../../../common/dsl.h"
#include "../../../common/thread_context.h"

#include "../../common/loader.h"
#include "gemm_mpp_core.h"

using namespace uzu::matmul;

#define PREFETCH_K_SIZE (((26624 / (128 * 2 * (int)sizeof(T))) / 16) * 16)
#define THREADGROUP_PADDING ((short)(16 / sizeof(T)))
#define THREADGROUP_LEADING_DIMENSION ((short)(PREFETCH_K_SIZE + THREADGROUP_PADDING))
#define THREADGROUP_TILE_SIZE (128 * THREADGROUP_LEADING_DIMENSION)

template <
    typename T,
    short BLOCK_ROWS,
    short BLOCK_COLS,
    short SIMDGROUPS_PER_ROW,
    short SIMDGROUPS_PER_COLUMN,
    short SUBTILE_ROWS,
    short SUBTILE_COLS,
    short MATMUL_K_STEP>
METAL_FUNC void gemm_mpp_impl(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant GemmParams* params,
    const bool align_m,
    const bool align_n,
    threadgroup T* left_shared,
    threadgroup T* right_shared,
    uint simd_group_id,
    uint simd_lane_id,
    uint2 threadgroup_position
) {
  constexpr short THREADGROUP_SIZE = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;

  constexpr short PREFETCH_K = short(PREFETCH_K_SIZE);
  constexpr short THREADGROUP_LD = PREFETCH_K + short(16 / sizeof(T));

  const int swizzle_size = pow2(params->swizzle_log);
  const int tid_y = ((threadgroup_position.y) * swizzle_size) +
                    ((threadgroup_position.x) % swizzle_size);
  const int tid_x = (threadgroup_position.x) / swizzle_size;

  if (params->threadgroups_per_row <= tid_x || params->threadgroups_per_column <= tid_y) {
    return;
  }

  threadgroup_barrier(mem_flags::mem_none);

  const int block_row_start = tid_y * BLOCK_ROWS;
  const int block_col_start = tid_x * BLOCK_COLS;
  const size_t block_row_start_long = size_t(block_row_start);
  const size_t block_col_start_long = size_t(block_col_start);

  const device T* left_block_ptr = left_matrix + block_row_start_long * params->leading_dimension_a;
  const device T* right_block_ptr = right_matrix + block_col_start_long * params->leading_dimension_b;

  ThreadgroupLoader<T, BLOCK_ROWS, PREFETCH_K, THREADGROUP_LD, 1, THREADGROUP_SIZE> loader_a(
      left_block_ptr, params->leading_dimension_a, left_shared, ushort(simd_group_id), ushort(simd_lane_id));
  ThreadgroupLoader<T, BLOCK_COLS, PREFETCH_K, THREADGROUP_LD, 1, THREADGROUP_SIZE> loader_b(
      right_block_ptr, params->leading_dimension_b, right_shared, ushort(simd_group_id), ushort(simd_lane_id));

  gemm_mpp_core<T, decltype(loader_a), decltype(loader_b),
                  BLOCK_ROWS, BLOCK_COLS, PREFETCH_K, THREADGROUP_LD,
                  SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN,
                  SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP>(
      loader_a, loader_b,
      output_matrix, params,
      align_m, align_n,
      block_row_start, block_col_start,
      left_shared, right_shared,
      simd_group_id, simd_lane_id);
}

#define GEMM_MPP_DISPATCH(T, BLOCK_ROWS, BLOCK_COLS, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN) \
  if (block_rows == BLOCK_ROWS && block_cols == BLOCK_COLS && \
      simdgroups_per_row == SIMDGROUPS_PER_ROW && simdgroups_per_column == SIMDGROUPS_PER_COLUMN) { \
    gemm_mpp_impl<T, BLOCK_ROWS, BLOCK_COLS, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, 16, 32, 16>( \
        left_matrix, right_matrix, output_matrix, params, \
        align_m, align_n, \
        left_shared, right_shared, \
        thread_context.threadgroup_index, thread_context.simdgroup_index, \
        uint2(group_x, group_y)); \
    return; \
  }

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemmMpp)(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    threadgroup T left_shared[THREADGROUP_TILE_SIZE],
    threadgroup T right_shared[THREADGROUP_TILE_SIZE],
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint simdgroups_per_row SPECIALIZE,
    const uint simdgroups_per_column SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(4),
    const uint thread_z THREADS(4),
    const ThreadContext thread_context
) {
  if (thread_context.threadgroup_index >= simdgroups_per_row * simdgroups_per_column) {
    return;
  }

  GEMM_MPP_DISPATCH(T, 64, 64, 2, 2)
  GEMM_MPP_DISPATCH(T, 32, 64, 2, 2)
  GEMM_MPP_DISPATCH(T, 64, 32, 4, 1)
}

#undef GEMM_MPP_DISPATCH

template <
    typename T,
    short BLOCK_ROWS,
    short BLOCK_COLS,
    short SIMDGROUPS_PER_ROW,
    short SIMDGROUPS_PER_COLUMN,
    short SUBTILE_ROWS,
    short SUBTILE_COLS,
    short MATMUL_K_STEP>
METAL_FUNC void gemm_mpp_native_impl(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant GemmParams* params,
    const bool align_m,
    const bool align_n,
    threadgroup T* left_shared,
    threadgroup T* right_shared,
    uint simd_group_id,
    uint simd_lane_id,
    uint2 threadgroup_position
) {
  constexpr short THREADGROUP_SIZE = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;

  constexpr short PREFETCH_K = short(PREFETCH_K_SIZE);
  constexpr short THREADGROUP_LD = PREFETCH_K + short(16 / sizeof(T));

  const int swizzle_size = pow2(params->swizzle_log);
  const int tid_y = ((threadgroup_position.y) * swizzle_size) +
                    ((threadgroup_position.x) % swizzle_size);
  const int tid_x = (threadgroup_position.x) / swizzle_size;

  if (params->threadgroups_per_row <= tid_x || params->threadgroups_per_column <= tid_y) {
    return;
  }

  threadgroup_barrier(mem_flags::mem_none);

  const int block_row_start = tid_y * BLOCK_ROWS;
  const int block_col_start = tid_x * BLOCK_COLS;
  const size_t block_row_start_long = size_t(block_row_start);
  const size_t block_col_start_long = size_t(block_col_start);

  const device T* left_block_ptr = left_matrix + block_row_start_long * params->leading_dimension_a;
  const device T* right_block_ptr = right_matrix + block_col_start_long * params->leading_dimension_b;

  ThreadgroupLoader<T, BLOCK_ROWS, PREFETCH_K, THREADGROUP_LD, 1, THREADGROUP_SIZE> loader_a(
      left_block_ptr, params->leading_dimension_a, left_shared, ushort(simd_group_id), ushort(simd_lane_id));
  ThreadgroupLoader<T, BLOCK_COLS, PREFETCH_K, THREADGROUP_LD, 1, THREADGROUP_SIZE> loader_b(
      right_block_ptr, params->leading_dimension_b, right_shared, ushort(simd_group_id), ushort(simd_lane_id));

  gemm_mpp_core<T, decltype(loader_a), decltype(loader_b),
                  BLOCK_ROWS, BLOCK_COLS, PREFETCH_K, THREADGROUP_LD,
                  SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN,
                  SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP, true>(
      loader_a, loader_b,
      output_matrix, params,
      align_m, align_n,
      block_row_start, block_col_start,
      left_shared, right_shared,
      simd_group_id, simd_lane_id);
}

#define GEMM_MPP_NATIVE_DISPATCH(T, BLOCK_ROWS, BLOCK_COLS, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN) \
  if (block_rows == BLOCK_ROWS && block_cols == BLOCK_COLS && \
      simdgroups_per_row == SIMDGROUPS_PER_ROW && simdgroups_per_column == SIMDGROUPS_PER_COLUMN) { \
    gemm_mpp_native_impl<T, BLOCK_ROWS, BLOCK_COLS, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, 16, 32, 16>( \
        left_matrix, right_matrix, output_matrix, params, \
        align_m, align_n, \
        left_shared, right_shared, \
        thread_context.threadgroup_index, thread_context.simdgroup_index, \
        uint2(group_x, group_y)); \
    return; \
  }

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemmMppNative)(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    threadgroup T left_shared[THREADGROUP_TILE_SIZE],
    threadgroup T right_shared[THREADGROUP_TILE_SIZE],
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint simdgroups_per_row SPECIALIZE,
    const uint simdgroups_per_column SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(4),
    const uint thread_z THREADS(4),
    const ThreadContext thread_context
) {
  if (thread_context.threadgroup_index >= simdgroups_per_row * simdgroups_per_column) {
    return;
  }

  GEMM_MPP_NATIVE_DISPATCH(T, 64, 64, 2, 2)
  GEMM_MPP_NATIVE_DISPATCH(T, 32, 64, 2, 2)
  GEMM_MPP_NATIVE_DISPATCH(T, 64, 32, 4, 1)
}

#undef GEMM_MPP_NATIVE_DISPATCH
// clang-format on

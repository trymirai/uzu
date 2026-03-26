// clang-format off
#include "../../../common/utils.h"
#include "../../../common/dsl.h"
#include "../../../common/thread_context.h"

#include "mxu_gemm_loop.h"

using namespace uzu::matmul;

template <
    typename T,
    short BLOCK_ROWS,
    short BLOCK_COLS,
    short SIMDGROUPS_PER_ROW,
    short SIMDGROUPS_PER_COLUMN>
METAL_FUNC void gemm_mpp_direct_impl(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant GemmParams* params,
    const bool align_m,
    const bool align_n,
    const bool align_k,
    uint simd_group_id,
    uint2 threadgroup_position
) {
  using AccumulatorType = float;

  constexpr short SIMDGROUP_ROWS = BLOCK_ROWS / SIMDGROUPS_PER_ROW;
  constexpr short SIMDGROUP_COLS = BLOCK_COLS / SIMDGROUPS_PER_COLUMN;

  const int swizzle_size = pow2(params->swizzle_log);
  const int tid_y = ((threadgroup_position.y) * swizzle_size) +
                    ((threadgroup_position.x) % swizzle_size);
  const int tid_x = (threadgroup_position.x) / swizzle_size;

  if (params->threadgroups_per_row <= tid_x || params->threadgroups_per_column <= tid_y) {
    return;
  }

  const int block_row_start = tid_y * BLOCK_ROWS;
  const int block_col_start = tid_x * BLOCK_COLS;
  const size_t block_row_start_long = size_t(block_row_start);
  const size_t block_col_start_long = size_t(block_col_start);

  const short simdgroup_row_offset = SIMDGROUP_ROWS * (simd_group_id / SIMDGROUPS_PER_COLUMN);
  const short simdgroup_col_offset = SIMDGROUP_COLS * (simd_group_id % SIMDGROUPS_PER_COLUMN);

  const device T* A = left_matrix + (block_row_start_long + simdgroup_row_offset) * params->leading_dimension_a;
  const device T* B = right_matrix + (block_col_start_long + simdgroup_col_offset) * params->leading_dimension_b;

  const int simdgroup_limit_m_int =
      align_m ? int(SIMDGROUP_ROWS) : min(int(SIMDGROUP_ROWS), params->M - (block_row_start + simdgroup_row_offset));
  const short simdgroup_limit_m = short(simdgroup_limit_m_int);
  const bool is_unaligned_m = align_m ? false : (simdgroup_limit_m != SIMDGROUP_ROWS);

  const int simdgroup_limit_n_int =
      align_n ? int(SIMDGROUP_COLS) : min(int(SIMDGROUP_COLS), params->N - (block_col_start + simdgroup_col_offset));
  const short simdgroup_limit_n = short(simdgroup_limit_n_int);
  const bool is_unaligned_n = align_n ? false : (simdgroup_limit_n != SIMDGROUP_COLS);

  auto result = [&] {
    if (align_k && !is_unaligned_m && !is_unaligned_n) {
      return mxu_gemm_loop<T, SIMDGROUP_ROWS, SIMDGROUP_COLS, false, true, true, true, true, AccumulatorType>(
          A, B, params->leading_dimension_a, params->leading_dimension_b,
          params->K, simdgroup_limit_m, simdgroup_limit_n);
    } else if (!is_unaligned_m && !is_unaligned_n) {
      return mxu_gemm_loop<T, SIMDGROUP_ROWS, SIMDGROUP_COLS, false, true, true, true, false, AccumulatorType>(
          A, B, params->leading_dimension_a, params->leading_dimension_b,
          params->K, simdgroup_limit_m, simdgroup_limit_n);
    } else {
      return mxu_gemm_loop<T, SIMDGROUP_ROWS, SIMDGROUP_COLS, false, true, false, false, false, AccumulatorType>(
          A, B, params->leading_dimension_a, params->leading_dimension_b,
          params->K, simdgroup_limit_m, simdgroup_limit_n);
    }
  }();

  device T* D = output_matrix +
      (block_row_start_long + simdgroup_row_offset) * params->leading_dimension_d +
      block_col_start_long + simdgroup_col_offset;

  if (!is_unaligned_m && !is_unaligned_n) {
    result.store(D, int(params->leading_dimension_d));
  } else {
    result.store_checked(D, int(params->leading_dimension_d), short2(simdgroup_limit_n, simdgroup_limit_m));
  }
}

#define GEMM_MPP_DIRECT_DISPATCH(T, BLOCK_ROWS, BLOCK_COLS, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN) \
  if (block_rows == BLOCK_ROWS && block_cols == BLOCK_COLS && \
      simdgroups_per_row == SIMDGROUPS_PER_ROW && simdgroups_per_column == SIMDGROUPS_PER_COLUMN) { \
    gemm_mpp_direct_impl<T, BLOCK_ROWS, BLOCK_COLS, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN>( \
        left_matrix, right_matrix, output_matrix, params, \
        align_m, align_n, align_k, \
        thread_context.threadgroup_index, \
        uint2(group_x, group_y)); \
    return; \
  }

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemmMppDirect)(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint simdgroups_per_row SPECIALIZE,
    const uint simdgroups_per_column SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
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

  GEMM_MPP_DIRECT_DISPATCH(T, 64, 64, 2, 2)
  GEMM_MPP_DIRECT_DISPATCH(T, 128, 128, 4, 4)
}

#undef GEMM_MPP_DIRECT_DISPATCH
// clang-format on

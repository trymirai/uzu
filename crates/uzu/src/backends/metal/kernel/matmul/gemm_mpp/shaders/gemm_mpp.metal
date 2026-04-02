// clang-format off
#include "../../../common/defines.h"
#include "../../../common/dsl.h"
#include "../../../common/thread_context.h"

#include "../../common/loader.h"
#include "gemm_mpp_core.h"

using namespace uzu::matmul;

#define PREFETCH_K_SIZE 32
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
    short MATMUL_K_STEP,
    bool ALIGNED_M,
    bool ALIGNED_N>
METAL_FUNC void gemm_mpp_impl(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant GemmParams* params,
    const bool align_m_rt,
    const bool align_n_rt,
    threadgroup T* left_shared,
    threadgroup T* right_shared,
    uint simd_group_id,
    uint simd_lane_id,
    uint2 threadgroup_position
) {
  constexpr short THREADGROUP_SIZE = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;
  constexpr short PREFETCH_K = short(PREFETCH_K_SIZE);
  constexpr short THREADGROUP_LD = PREFETCH_K + short(16 / sizeof(T));

  int tid_x, tid_y;
  if (threadgroup_position.y == 0 && threadgroup_position.x >= uint(params->threadgroups_per_row)) {
    uint linear_id = threadgroup_position.x;
    uint mx = linear_id;
    uint my = linear_id >> 1;
    mx &= 0x55555555u; mx = (mx | (mx >> 1)) & 0x33333333u;
    mx = (mx | (mx >> 2)) & 0x0F0F0F0Fu; mx = (mx | (mx >> 4)) & 0x00FF00FFu;
    mx = (mx | (mx >> 8)) & 0x0000FFFFu;
    my &= 0x55555555u; my = (my | (my >> 1)) & 0x33333333u;
    my = (my | (my >> 2)) & 0x0F0F0F0Fu; my = (my | (my >> 4)) & 0x00FF00FFu;
    my = (my | (my >> 8)) & 0x0000FFFFu;
    tid_x = int(mx);
    tid_y = int(my);
  } else {
    tid_x = int(threadgroup_position.x);
    tid_y = int(threadgroup_position.y);
  }

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
      align_m_rt, align_n_rt,
      block_row_start, block_col_start,
      left_shared, right_shared,
      simd_group_id, simd_lane_id);
}

#define GEMM_MPP_DISPATCH(T, BLOCK_ROWS, BLOCK_COLS, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN) \
  if (block_rows == BLOCK_ROWS && block_cols == BLOCK_COLS && \
      simdgroups_per_row == SIMDGROUPS_PER_ROW && simdgroups_per_column == SIMDGROUPS_PER_COLUMN) { \
    gemm_mpp_impl<T, BLOCK_ROWS, BLOCK_COLS, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, 16, 32, 16, false, false>( \
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
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    const ThreadContext thread_context
) {
  GEMM_MPP_DISPATCH(T, 64, 64, 2, 2)
  GEMM_MPP_DISPATCH(T, 32, 64, 2, 2)
  GEMM_MPP_DISPATCH(T, 64, 32, 4, 1)
}

#undef GEMM_MPP_DISPATCH
// clang-format on

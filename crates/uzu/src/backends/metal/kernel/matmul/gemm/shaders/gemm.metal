// clang-format off
#include "../../../common/utils.h"
#include "../../../common/dsl.h"
#include "../../../common/thread_context.h"

#include "gemm.h"

using namespace uzu::matmul;

template <typename T, int BLOCK_ROWS, int BLOCK_COLS, int BLOCK_DEPTH, int SIMDGROUPS_PER_ROW, int SIMDGROUPS_PER_COLUMN, bool MN_aligned, bool K_aligned>
METAL_FUNC void gemm_dispatch(
    const device T* a,
    const device T* b,
    device T* d,
    const constant GemmParams* params,
    threadgroup T* a_shared,
    threadgroup T* b_shared,
    uint simd_lane_id,
    uint simd_group_id,
    uint2 tid,
    uint3 lid
) {
  using ThreadgroupGemmType = ThreadgroupGemm<T, T, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, false, true, MN_aligned, K_aligned, float>;
  ThreadgroupGemmType::run(a, b, d, params, a_shared, b_shared, simd_lane_id, simd_group_id, tid, lid);
}

#define GEMM_DISPATCH(T, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN) \
  if (block_rows == BLOCK_ROWS && block_cols == BLOCK_COLS && block_depth == BLOCK_DEPTH && \
      simdgroups_per_row == SIMDGROUPS_PER_ROW && simdgroups_per_column == SIMDGROUPS_PER_COLUMN) { \
    if (align_m && align_n) { \
      if (align_k) \
        gemm_dispatch<T, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, true, true>(a, b, d, params, a_shared, b_shared, thread_context.simdgroup_index, thread_context.threadgroup_index, uint2(group_x, group_y), uint3(thread_x, thread_y, thread_z)); \
      else \
        gemm_dispatch<T, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, true, false>(a, b, d, params, a_shared, b_shared, thread_context.simdgroup_index, thread_context.threadgroup_index, uint2(group_x, group_y), uint3(thread_x, thread_y, thread_z)); \
    } else { \
      if (align_k) \
        gemm_dispatch<T, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, false, true>(a, b, d, params, a_shared, b_shared, thread_context.simdgroup_index, thread_context.threadgroup_index, uint2(group_x, group_y), uint3(thread_x, thread_y, thread_z)); \
      else \
        gemm_dispatch<T, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, false, false>(a, b, d, params, a_shared, b_shared, thread_context.simdgroup_index, thread_context.threadgroup_index, uint2(group_x, group_y), uint3(thread_x, thread_y, thread_z)); \
    } \
    return; \
  }

#define GEMM_MAX_THREADGROUP_A 2560
#define GEMM_MAX_THREADGROUP_B 1536

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemm)(
    const device T* a,
    const device T* b,
    device T* d,
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    threadgroup T a_shared[GEMM_MAX_THREADGROUP_A],
    threadgroup T b_shared[GEMM_MAX_THREADGROUP_B],
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint block_depth SPECIALIZE,
    const uint simdgroups_per_row SPECIALIZE,
    const uint simdgroups_per_column SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    const ThreadContext thread_context
) {
  GEMM_DISPATCH(T, 64, 64, 16, 2, 2)
  GEMM_DISPATCH(T, 64, 64, 16, 1, 2)
  GEMM_DISPATCH(T, 64, 32, 32, 2, 2)
  GEMM_DISPATCH(T, 32, 64, 16, 1, 2)
  GEMM_DISPATCH(T, 32, 64, 16, 2, 2)
  GEMM_DISPATCH(T, 32, 32, 16, 2, 2)
  GEMM_DISPATCH(T, 64, 32,  8, 4, 1)
}

// clang-format on

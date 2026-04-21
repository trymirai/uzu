#include "../common/dsl.h"
#include "../common/thread_context.h"

#include "common/gemm_mpp_core.h"

using namespace uzu::matmul;

#define GEMM_MPP_THREADGROUP_TILE_SIZE (128 * (32 + (short)(16 / sizeof(T))))

template <typename T, uint BLOCK_ROWS, uint BLOCK_COLS, uint SIMDGROUPS_PER_ROW, uint SIMDGROUPS_PER_COLUMN>
VARIANTS(T, float, half, bfloat)
VARIANTS(BLOCK_ROWS, 32, 64)
VARIANTS(BLOCK_COLS, 32, 64)
VARIANTS(SIMDGROUPS_PER_ROW, 2, 4)
VARIANTS(SIMDGROUPS_PER_COLUMN, 1, 2)
CONSTRAINT(BLOCK_ROWS >= 16 * SIMDGROUPS_PER_ROW && BLOCK_COLS >= 32 * SIMDGROUPS_PER_COLUMN)
KERNEL(MatmulGemmMpp)(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant uzu::matmul::GemmParams& params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant float& ab_scale,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool is_accumulate SPECIALIZE,
    threadgroup T left_shared[GEMM_MPP_THREADGROUP_TILE_SIZE],
    threadgroup T right_shared[GEMM_MPP_THREADGROUP_TILE_SIZE],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    const ThreadContext thread_context
) {
  ThreadgroupGemmMpp<
      T,
      BLOCK_ROWS,
      BLOCK_COLS,
      SIMDGROUPS_PER_ROW,
      SIMDGROUPS_PER_COLUMN>::
      run(left_matrix,
          right_matrix,
          output_matrix,
          params,
          align_m,
          align_n,
          is_accumulate,
          ab_scale,
          left_shared,
          right_shared,
          thread_context.simdgroup_index,
          thread_context.threadgroup_index,
          uint2(group_x, group_y));
}

#include "../common/dsl.h"
#include "../common/thread_context.h"

#include "common/gemm.h"

using namespace uzu::matmul;

#define GEMM_MAX_THREADGROUP_A 2560
#define GEMM_MAX_THREADGROUP_B 1536

template <typename T, uint BLOCK_ROWS, uint BLOCK_COLS, uint BLOCK_DEPTH, uint SIMDGROUPS_PER_ROW, uint SIMDGROUPS_PER_COLUMN, bool MN_ALIGNED, bool K_ALIGNED>
VARIANTS(T, float, half, bfloat)
VARIANTS(BLOCK_ROWS, 32, 64)
VARIANTS(BLOCK_COLS, 32, 64)
VARIANTS(BLOCK_DEPTH, 8, 16, 32)
VARIANTS(SIMDGROUPS_PER_ROW, 1, 2, 4)
VARIANTS(SIMDGROUPS_PER_COLUMN, 1, 2)
VARIANTS(MN_ALIGNED, false, true)
VARIANTS(K_ALIGNED, false, true)
CONSTRAINT(max(BLOCK_ROWS, BLOCK_COLS) <= 32 * SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN)
KERNEL(MatmulGemm)(
    const device T* a,
    const device T* b,
    device T* d,
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant float& ab_scale,
    const bool is_accumulate SPECIALIZE,
    threadgroup T a_shared[GEMM_MAX_THREADGROUP_A],
    threadgroup T b_shared[GEMM_MAX_THREADGROUP_B],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    const ThreadContext thread_context
) {
  ThreadgroupGemm<
      T,
      T,
      BLOCK_ROWS,
      BLOCK_COLS,
      BLOCK_DEPTH,
      SIMDGROUPS_PER_ROW,
      SIMDGROUPS_PER_COLUMN,
      false, // transpose_a
      true,  // transpose_b
      MN_ALIGNED,
      K_ALIGNED,
      float>::
      run(a,
          b,
          d,
          params,
          ab_scale,
          is_accumulate,
          a_shared,
          b_shared,
          thread_context,
          uint2(group_x, group_y),
          uint3(thread_x, thread_y, thread_z));
}

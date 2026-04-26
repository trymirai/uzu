#include "../common/dsl.h"

#include "common/gemm_mpp_core.h"

using namespace uzu;
using namespace uzu::matmul;

template <typename T, uint BLOCK_ROWS, uint BLOCK_COLS, uint SIMDGROUPS_PER_ROW, uint SIMDGROUPS_PER_COLUMN>
VARIANTS(T, float, half, bfloat)
VARIANTS(BLOCK_ROWS, 32, 64, 128)
VARIANTS(BLOCK_COLS, 32, 64, 128)
VARIANTS(SIMDGROUPS_PER_ROW, 2, 4)
VARIANTS(SIMDGROUPS_PER_COLUMN, 1, 2, 4)
CONSTRAINT(
  (BLOCK_ROWS == 32  && BLOCK_COLS == 64  && SIMDGROUPS_PER_ROW == 2 && SIMDGROUPS_PER_COLUMN == 2) ||
  (BLOCK_ROWS == 64  && BLOCK_COLS == 32  && SIMDGROUPS_PER_ROW == 4 && SIMDGROUPS_PER_COLUMN == 1) ||
  (BLOCK_ROWS == 64  && BLOCK_COLS == 64  && SIMDGROUPS_PER_ROW == 2 && SIMDGROUPS_PER_COLUMN == 2) ||
  (BLOCK_ROWS == 128 && BLOCK_COLS == 128 && SIMDGROUPS_PER_ROW == 4 && SIMDGROUPS_PER_COLUMN == 4))
KERNEL(MatmulGemmMpp)(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant float& ab_scale,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const bool apply_ab_scale SPECIALIZE,
    const bool is_accumulate SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(SIMDGROUPS_PER_ROW),
    const uint thread_z THREADS(SIMDGROUPS_PER_COLUMN)
) {
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;
  dispatch_bool(align_m, [&](auto align_m_enabled) {
    dispatch_bool(align_n, [&](auto align_n_enabled) {
      dispatch_bool(align_k, [&](auto align_k_enabled) {
        dispatch_bool(apply_ab_scale, [&](auto scale_enabled) {
          dispatch_bool(is_accumulate, [&](auto accumulate_enabled) {
            GemmMppCore<
                T,
                BLOCK_ROWS,
                BLOCK_COLS,
                SIMDGROUPS_PER_ROW,
                SIMDGROUPS_PER_COLUMN,
                align_m_enabled.value,
                align_n_enabled.value,
                align_k_enabled.value,
                scale_enabled.value,
                accumulate_enabled.value>::
                run(left_matrix,
                    right_matrix,
                    output_matrix,
                    params,
                    ab_scale,
                    uint2(group_x, group_y));
          });
        });
      });
    });
  });
}

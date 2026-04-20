#pragma once

#include "../../common/defines.h"
#include "../../common/integral_constant.h"
using namespace uzu;
#include "mxu_matmul.h"
#include "mxu_gemm_loop.h"
#include "../../generated/matmul.h"

using namespace metal;

namespace uzu {
namespace matmul {

template <
    typename T,
    ushort BLOCK_ROWS,
    ushort BLOCK_COLS,
    ushort SIMDGROUPS_PER_ROW,
    ushort SIMDGROUPS_PER_COLUMN,
    bool APPLY_AB_SCALE,
    bool IS_ACCUMULATE>
struct GemmMppCore {
  METAL_CONST ushort SIMDGROUP_BLOCK_M = BLOCK_ROWS / SIMDGROUPS_PER_ROW;
  METAL_CONST ushort SIMDGROUP_BLOCK_N = BLOCK_COLS / SIMDGROUPS_PER_COLUMN;

  METAL_CONST ushort SIMDGROUP_BLOCK_K = 32;
  METAL_CONST ushort BLOCK_K = 256;

  METAL_CONST ushort TILES_M = SIMDGROUP_BLOCK_M / MxuFragment::FRAGMENT_ROWS;
  METAL_CONST ushort TILES_N = SIMDGROUP_BLOCK_N / MxuFragment::FRAGMENT_COLS;

  using AccumulatorType = float;

  static METAL_FUNC void run(
      const device T* left_matrix,
      const device T* right_matrix,
      device T* output_matrix,
      const constant GemmParams* params,
      const bool align_m,
      const bool align_n,
      const bool align_k,
      const float ab_scale,
      uint simd_group_id,
      uint2 threadgroup_position,
      const thread ThreadContext& thread_context
  ) {
    uint tile_id_x, tile_id_y;
    if (params->use_morton) {
      uint linear_id = threadgroup_position.x;
      uint morton_x = linear_id;
      uint morton_y = linear_id >> 1;
      morton_x &= 0x55555555u;
      morton_x = (morton_x | (morton_x >> 1)) & 0x33333333u;
      morton_x = (morton_x | (morton_x >> 2)) & 0x0F0F0F0Fu;
      morton_x = (morton_x | (morton_x >> 4)) & 0x00FF00FFu;
      morton_x = (morton_x | (morton_x >> 8)) & 0x0000FFFFu;
      morton_y &= 0x55555555u;
      morton_y = (morton_y | (morton_y >> 1)) & 0x33333333u;
      morton_y = (morton_y | (morton_y >> 2)) & 0x0F0F0F0Fu;
      morton_y = (morton_y | (morton_y >> 4)) & 0x00FF00FFu;
      morton_y = (morton_y | (morton_y >> 8)) & 0x0000FFFFu;
      tile_id_x = morton_x;
      tile_id_y = morton_y;
    } else {
      tile_id_x = threadgroup_position.x;
      tile_id_y = threadgroup_position.y;
    }

    if (tile_id_x >= params->threadgroups_per_row ||
        tile_id_y >= params->threadgroups_per_column) {
      return;
    }

    const uint block_row_start = tile_id_y * BLOCK_ROWS;
    const uint block_col_start = tile_id_x * BLOCK_COLS;
    const size_t block_row_start_long = size_t(block_row_start);
    const size_t block_col_start_long = size_t(block_col_start);

    const device T* left_block_ptr =
        left_matrix + block_row_start_long * params->leading_dimension_a;
    const device T* right_block_ptr =
        right_matrix + block_col_start_long * params->leading_dimension_b;

    const ushort tile_row_offset =
        SIMDGROUP_BLOCK_M * (simd_group_id / SIMDGROUPS_PER_COLUMN);
    const ushort tile_col_offset =
        SIMDGROUP_BLOCK_N * (simd_group_id % SIMDGROUPS_PER_COLUMN);

    device T* output_ptr =
        output_matrix + block_row_start_long * params->leading_dimension_d +
        block_col_start_long + tile_row_offset * params->leading_dimension_d +
        tile_col_offset;

    const short simdgroup_limit_m =
        align_m
            ? SIMDGROUP_BLOCK_M
            : short(
                  min(int(SIMDGROUP_BLOCK_M),
                      int(params->M) - int(block_row_start + tile_row_offset))
              );
    const short simdgroup_limit_n =
        align_n
            ? SIMDGROUP_BLOCK_N
            : short(
                  min(int(SIMDGROUP_BLOCK_N),
                      int(params->N) - int(block_col_start + tile_col_offset))
              );

    const device T* left_simdgroup_ptr =
        left_block_ptr + size_t(tile_row_offset) * params->leading_dimension_a;
    const device T* right_simdgroup_ptr =
        right_block_ptr +
        size_t(tile_col_offset) * int(params->leading_dimension_b);

    const int aligned_k_iterations = int(params->K) / int(BLOCK_K);

    dispatch_bool(align_k, [&](auto aligned_k) {
      dispatch_bool(
          align_m || (simdgroup_limit_m == SIMDGROUP_BLOCK_M),
          [&](auto aligned_m) {
            dispatch_bool(
                align_n || (simdgroup_limit_n == SIMDGROUP_BLOCK_N),
                [&](auto aligned_n) {
                  auto accumulator_tile = gemm_loop<
                      T,
                      SIMDGROUP_BLOCK_M,
                      SIMDGROUP_BLOCK_N,
                      SIMDGROUP_BLOCK_K,
                      BLOCK_K,
                      false,
                      true,
                      aligned_m.value,
                      aligned_n.value,
                      aligned_k.value,
                      AccumulatorType>(
                      left_simdgroup_ptr,
                      right_simdgroup_ptr,
                      int(params->leading_dimension_a),
                      int(params->leading_dimension_b),
                      int(params->K),
                      aligned_k_iterations,
                      simdgroup_limit_m,
                      simdgroup_limit_n,
                      thread_context
                  );

                  if constexpr (APPLY_AB_SCALE) {
                    METAL_PRAGMA_UNROLL
                    for (ushort i = 0; i < accumulator_tile.ELEMENTS_PER_TILE;
                         i++) {
                      accumulator_tile.elems()[i] *= AccumulatorType(ab_scale);
                    }
                  }

                  if constexpr (IS_ACCUMULATE) {
                    MxuTile<T, TILES_M, TILES_N> existing_output(
                        thread_context
                    );
                    if constexpr (aligned_m.value && aligned_n.value) {
                      existing_output.load(
                          output_ptr,
                          int(params->leading_dimension_d)
                      );
                    } else {
                      existing_output.load_safe(
                          output_ptr,
                          int(params->leading_dimension_d),
                          short2(simdgroup_limit_n, simdgroup_limit_m)
                      );
                    }
                    METAL_PRAGMA_UNROLL
                    for (ushort i = 0; i < accumulator_tile.ELEMENTS_PER_TILE;
                         i++) {
                      accumulator_tile.elems()[i] +=
                          AccumulatorType(existing_output.elems()[i]);
                    }
                  }

                  if constexpr (aligned_m.value && aligned_n.value) {
                    accumulator_tile.store(
                        output_ptr,
                        int(params->leading_dimension_d)
                    );
                  } else {
                    accumulator_tile.store_safe(
                        output_ptr,
                        int(params->leading_dimension_d),
                        short2(simdgroup_limit_n, simdgroup_limit_m)
                    );
                  }
                }
            );
          }
      );
    });
  }
};

} // namespace matmul
} // namespace uzu

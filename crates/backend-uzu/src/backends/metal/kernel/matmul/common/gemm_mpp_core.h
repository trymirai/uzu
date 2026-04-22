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

  METAL_CONST ushort TILES_M =
      SIMDGROUP_BLOCK_M / MxuFragmentOps::FRAGMENT_ROWS;
  METAL_CONST ushort TILES_N =
      SIMDGROUP_BLOCK_N / MxuFragmentOps::FRAGMENT_COLS;

  using AccumulatorType = float;

  static METAL_FUNC void run(
      const device T* left_matrix,
      const device T* right_matrix,
      device T* output_matrix,
      const constant GemmParams& params,
      const bool align_m,
      const bool align_n,
      const bool align_k,
      const float ab_scale,
      uint simd_group_id,
      uint2 threadgroup_position,
      const thread ThreadContext& thread_context
  ) {
    uint tile_id_x, tile_id_y;
    if (params.use_morton) {
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

    if (tile_id_x >= params.threadgroups_per_row ||
        tile_id_y >= params.threadgroups_per_column) {
      return;
    }

    const uint block_row_start = tile_id_y * BLOCK_ROWS;
    const uint block_col_start = tile_id_x * BLOCK_COLS;
    const size_t block_row_start_long = size_t(block_row_start);
    const size_t block_col_start_long = size_t(block_col_start);

    const device T* left_block_ptr =
        left_matrix + block_row_start_long * params.leading_dimension_a;
    const device T* right_block_ptr =
        right_matrix + block_col_start_long * params.leading_dimension_b;

    ThreadgroupLoader<
        T,
        BLOCK_ROWS,
        PREFETCH_K,
        THREADGROUP_LD,
        1,
        THREADGROUP_SIZE>
        left_loader(
            left_block_ptr,
            params.leading_dimension_a,
            left_shared,
            ushort(simd_group_id),
            ushort(simd_lane_id)
        );
    ThreadgroupLoader<
        T,
        BLOCK_COLS,
        PREFETCH_K,
        THREADGROUP_LD,
        1,
        THREADGROUP_SIZE>
        right_loader(
            right_block_ptr,
            params.leading_dimension_b,
            right_shared,
            ushort(simd_group_id),
            ushort(simd_lane_id)
        );

    const ushort tile_row_offset =
        SIMDGROUP_BLOCK_M * (simd_group_id / SIMDGROUPS_PER_COLUMN);
    const ushort tile_col_offset =
        SIMDGROUP_BLOCK_N * (simd_group_id % SIMDGROUPS_PER_COLUMN);

    device T* output_ptr =
        output_matrix + block_row_start_long * params.leading_dimension_d +
        block_col_start_long + tile_row_offset * params.leading_dimension_d +
        tile_col_offset;

    const short simdgroup_limit_m =
        align_m
            ? SIMDGROUP_BLOCK_M
            : short(
                  min(int(SIMDGROUP_BLOCK_M),
                      int(params.M) - int(block_row_start + tile_row_offset))
              );
    const short simdgroup_limit_n =
        align_n
            ? SIMDGROUP_BLOCK_N
            : short(
                  min(int(SIMDGROUP_BLOCK_N),
                      int(params.N) - int(block_col_start + tile_col_offset))
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

    const uint leading_dimension_d = params.leading_dimension_d;

                  if constexpr (IS_ACCUMULATE) {
                    Fragment<T, TILES_M, TILES_N, MxuFragmentOps>
                        existing_output(thread_context);
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
                      accumulator_tile.elements()[i] +=
                          AccumulatorType(existing_output.elements()[i]);
                    }
                  }

    const uint full_prefetch_iterations = params.K / PREFETCH_K;
    const uint k_remainder = params.K - full_prefetch_iterations * PREFETCH_K;

    const ushort actual_block_rows =
        align_m ? BLOCK_ROWS
                : ushort(min(uint(BLOCK_ROWS), params.M - block_row_start));
    const ushort actual_block_cols =
        align_n ? BLOCK_COLS
                : ushort(min(uint(BLOCK_COLS), params.N - block_col_start));

    // Main loop
    for (uint outer_k = 0; outer_k < full_prefetch_iterations; outer_k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (align_m) {
        left_loader.load_unsafe();
      } else {
        left_loader.load_safe(short2(PREFETCH_K, actual_block_rows));
      }
      if (align_n) {
        right_loader.load_unsafe();
      } else {
        right_loader.load_safe(short2(PREFETCH_K, actual_block_cols));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      accumulate_tiles(
          INNER_K_STEPS,
          tile_row_offset,
          tile_col_offset,
          left_shared,
          right_shared,
          matmul_operation,
          left_tensor,
          right_tensor,
          accumulator_tensor,
          accumulator_storage
      );

      left_loader.next();
      right_loader.next();
    }

    // Remainder loop
    if (k_remainder > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      left_loader.load_safe(short2(k_remainder, actual_block_rows));
      right_loader.load_safe(short2(k_remainder, actual_block_cols));
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const short remainder_steps =
          short((k_remainder + MATMUL_K_STEP - 1) / MATMUL_K_STEP);

      accumulate_tiles(
          remainder_steps,
          tile_row_offset,
          tile_col_offset,
          left_shared,
          right_shared,
          matmul_operation,
          left_tensor,
          right_tensor,
          accumulator_tensor,
          accumulator_storage
      );
    }

    // Store results
    METAL_PRAGMA_UNROLL
    for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
      METAL_PRAGMA_UNROLL
      for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
        const short row_offset = tile_m * SUBTILE_ROWS;
        const short col_offset = tile_n * SUBTILE_COLS;
        const short m_limit =
            align_m ? SUBTILE_ROWS
                    : short(max(0, int(simdgroup_limit_m) - row_offset));
        const short n_limit =
            align_n ? SUBTILE_COLS
                    : short(max(0, int(simdgroup_limit_n) - col_offset));
        if (m_limit <= 0 || n_limit <= 0)
          continue;

        device T* output_tile_ptr =
            output_ptr + row_offset * leading_dimension_d + col_offset;

        METAL_PRAGMA_UNROLL
        for (short i = 0; i < short(ACCUMULATOR_CAPACITY); i++) {
          auto coord = accumulator_tensor.get_multidimensional_index(i);
          const bool is_valid_element = accumulator_tensor.is_valid_element(i);
          AccumulatorType accumulated_value =
              accumulator_storage[tile_m * TILES_N + tile_n][i] *
              AccumulatorType(ab_scale);
          if (align_m && align_n) {
            if (is_valid_element) {
              if (is_accumulate) {
                accumulated_value += AccumulatorType(
                    output_tile_ptr[coord[1] * leading_dimension_d + coord[0]]
                );
              }
              output_tile_ptr[coord[1] * leading_dimension_d + coord[0]] =
                  T(accumulated_value);
            }
          } else {
            if (is_valid_element && coord[1] < m_limit && coord[0] < n_limit) {
              if (is_accumulate) {
                accumulated_value += AccumulatorType(
                    output_tile_ptr[coord[1] * leading_dimension_d + coord[0]]
                );
              }
              output_tile_ptr[coord[1] * leading_dimension_d + coord[0]] =
                  T(accumulated_value);
            }
          }
      );
    });
  }
};

} // namespace matmul
} // namespace uzu

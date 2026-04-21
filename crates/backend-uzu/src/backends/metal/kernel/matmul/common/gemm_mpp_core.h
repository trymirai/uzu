#pragma once

#include "defines.h"
#include "loader.h"
#include "../../generated/matmul.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace uzu {
namespace matmul {

template <
    typename T,
    short BLOCK_ROWS,
    short BLOCK_COLS,
    short SIMDGROUPS_PER_ROW,
    short SIMDGROUPS_PER_COLUMN,
    short SUBTILE_ROWS = 16,
    short SUBTILE_COLS = 32,
    short MATMUL_K_STEP = 16,
    // Number of cooperative tensor elements owned by each thread.
    // Determined by the MPP compiler intrinsic get_capacity().
    // Empirically: SUBTILE_ROWS * SUBTILE_COLS / SIMD_SIZE = 16 * 32 / 32 = 16.
    short ACCUMULATOR_CAPACITY = 16>
struct ThreadgroupGemmMpp {
  METAL_CONST short PREFETCH_K = 32;
  METAL_CONST short THREADGROUP_PADDING = short(16 / sizeof(T));
  METAL_CONST short THREADGROUP_LD = PREFETCH_K + THREADGROUP_PADDING;
  METAL_CONST short THREADGROUP_SIZE =
      SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;
  METAL_CONST short SIMDGROUP_BLOCK_M = BLOCK_ROWS / SIMDGROUPS_PER_ROW;
  METAL_CONST short SIMDGROUP_BLOCK_N = BLOCK_COLS / SIMDGROUPS_PER_COLUMN;
  METAL_CONST short TILES_M = SIMDGROUP_BLOCK_M / SUBTILE_ROWS;
  METAL_CONST short TILES_N = SIMDGROUP_BLOCK_N / SUBTILE_COLS;
  METAL_CONST short INNER_K_STEPS = PREFETCH_K / MATMUL_K_STEP;

  using AccumulatorType = float;

  template <
      typename MatmulOperation,
      typename LeftTensor,
      typename RightTensor,
      typename AccumulatorTensor>
  static METAL_FUNC void accumulate_tiles(
      const short k_steps,
      const ushort tile_row_offset,
      const ushort tile_col_offset,
      threadgroup T* left_shared,
      threadgroup T* right_shared,
      thread MatmulOperation& matmul_operation,
      thread LeftTensor& left_tensor,
      thread RightTensor& right_tensor,
      thread AccumulatorTensor& accumulator_tensor,
      thread AccumulatorType
          accumulator_storage[TILES_M * TILES_N][ACCUMULATOR_CAPACITY]
  ) {
    METAL_PRAGMA_UNROLL
    for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
      METAL_PRAGMA_UNROLL
      for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
        const short subtile_index = tile_m * TILES_N + tile_n;
        const short left_row_base = tile_row_offset + tile_m * SUBTILE_ROWS;
        const short right_col_base = tile_col_offset + tile_n * SUBTILE_COLS;

        METAL_PRAGMA_UNROLL
        for (short i = 0; i < ACCUMULATOR_CAPACITY; i++) {
          accumulator_tensor[i] = accumulator_storage[subtile_index][i];
        }

        for (short k_step = 0; k_step < k_steps; k_step++) {
          const short k_offset = k_step * MATMUL_K_STEP;

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < left_tensor.get_capacity(); i++) {
            auto coord = left_tensor.get_multidimensional_index(i);
            left_tensor[i] = left_shared
                [(left_row_base + coord[1]) * THREADGROUP_LD + coord[0] +
                 k_offset];
          }

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < right_tensor.get_capacity(); i++) {
            auto coord = right_tensor.get_multidimensional_index(i);
            right_tensor[i] = right_shared
                [(right_col_base + coord[1]) * THREADGROUP_LD + coord[0] +
                 k_offset];
          }

          matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);
        }

        METAL_PRAGMA_UNROLL
        for (short i = 0; i < ACCUMULATOR_CAPACITY; i++) {
          accumulator_storage[subtile_index][i] = accumulator_tensor[i];
        }
      }
    }
  }

  static METAL_FUNC void run(
      const device T* left_matrix,
      const device T* right_matrix,
      device T* output_matrix,
      const constant GemmParams& params,
      const bool align_m,
      const bool align_n,
      const bool is_accumulate,
      const float ab_scale,
      threadgroup T* left_shared,
      threadgroup T* right_shared,
      uint simd_lane_id,
      uint simd_group_id,
      uint2 threadgroup_position
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

    constexpr auto matmul_descriptor = mpp::tensor_ops::matmul2d_descriptor(
        SUBTILE_ROWS,
        SUBTILE_COLS,
        MATMUL_K_STEP,
        false,
        true,
        false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );

    mpp::tensor_ops::matmul2d<matmul_descriptor, metal::execution_simdgroup>
        matmul_operation;

    auto left_tensor =
        matmul_operation.template get_left_input_cooperative_tensor<
            T,
            T,
            AccumulatorType>();
    auto right_tensor =
        matmul_operation.template get_right_input_cooperative_tensor<
            T,
            T,
            AccumulatorType>();
    auto accumulator_tensor =
        matmul_operation.template get_destination_cooperative_tensor<
            decltype(left_tensor),
            decltype(right_tensor),
            AccumulatorType>();

    const uint leading_dimension_d = params.leading_dimension_d;

    AccumulatorType accumulator_storage[TILES_M * TILES_N]
                                       [ACCUMULATOR_CAPACITY] = {{0}};

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
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu

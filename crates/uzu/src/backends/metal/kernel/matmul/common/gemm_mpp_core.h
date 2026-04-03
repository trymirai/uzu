#pragma once

#include "defines.h"
#include "loader.h"
#include "../../../generated/matmul.h"

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
    short MATMUL_K_STEP = 16>
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

  static METAL_FUNC void run(
      const device T* left_matrix,
      const device T* right_matrix,
      device T* output_matrix,
      const constant GemmParams* params,
      const bool align_m,
      const bool align_n,
      threadgroup T* left_shared,
      threadgroup T* right_shared,
      uint simd_lane_id,
      uint simd_group_id,
      uint2 threadgroup_position
  ) {
    int tid_x, tid_y;
    if (params->use_morton != 0) {
      uint linear_id = threadgroup_position.x;
      uint mx = linear_id;
      uint my = linear_id >> 1;
      mx &= 0x55555555u;
      mx = (mx | (mx >> 1)) & 0x33333333u;
      mx = (mx | (mx >> 2)) & 0x0F0F0F0Fu;
      mx = (mx | (mx >> 4)) & 0x00FF00FFu;
      mx = (mx | (mx >> 8)) & 0x0000FFFFu;
      my &= 0x55555555u;
      my = (my | (my >> 1)) & 0x33333333u;
      my = (my | (my >> 2)) & 0x0F0F0F0Fu;
      my = (my | (my >> 4)) & 0x00FF00FFu;
      my = (my | (my >> 8)) & 0x0000FFFFu;
      tid_x = int(mx);
      tid_y = int(my);
    } else {
      tid_x = int(threadgroup_position.x);
      tid_y = int(threadgroup_position.y);
    }

    if (tid_x >= int(params->threadgroups_per_row) ||
        tid_y >= int(params->threadgroups_per_column)) {
      return;
    }

    const int block_row_start = tid_y * BLOCK_ROWS;
    const int block_col_start = tid_x * BLOCK_COLS;
    const size_t block_row_start_long = size_t(block_row_start);
    const size_t block_col_start_long = size_t(block_col_start);

    const device T* left_block_ptr =
        left_matrix + block_row_start_long * params->leading_dimension_a;
    const device T* right_block_ptr =
        right_matrix + block_col_start_long * params->leading_dimension_b;

    ThreadgroupLoader<
        T,
        BLOCK_ROWS,
        PREFETCH_K,
        THREADGROUP_LD,
        1,
        THREADGROUP_SIZE>
        loader_a(
            left_block_ptr,
            params->leading_dimension_a,
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
        loader_b(
            right_block_ptr,
            params->leading_dimension_b,
            right_shared,
            ushort(simd_group_id),
            ushort(simd_lane_id)
        );

    // Setup MPP matmul operation
    using AccumulatorType = float;

    const short tile_row_offset =
        SIMDGROUP_BLOCK_M * (simd_group_id / SIMDGROUPS_PER_COLUMN);
    const short tile_col_offset =
        SIMDGROUP_BLOCK_N * (simd_group_id % SIMDGROUPS_PER_COLUMN);

    device T* output_ptr =
        output_matrix + block_row_start_long * params->leading_dimension_d +
        block_col_start_long + tile_row_offset * params->leading_dimension_d +
        tile_col_offset;

    const short simdgroup_limit_m =
        align_m ? SIMDGROUP_BLOCK_M
                : short(
                      min(int(SIMDGROUP_BLOCK_M),
                          int(params->M) - (block_row_start + tile_row_offset))
                  );
    const short simdgroup_limit_n =
        align_n ? SIMDGROUP_BLOCK_N
                : short(
                      min(int(SIMDGROUP_BLOCK_N),
                          int(params->N) - (block_col_start + tile_col_offset))
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

    const short accumulator_capacity = accumulator_tensor.get_capacity();

    AccumulatorType accum_storage[TILES_M * TILES_N][16];
    METAL_PRAGMA_UNROLL
    for (short t = 0; t < TILES_M * TILES_N; t++) {
      METAL_PRAGMA_UNROLL
      for (short i = 0; i < 16; i++) {
        accum_storage[t][i] = AccumulatorType(0);
      }
    }

    const int full_prefetch_iterations = params->K / PREFETCH_K;
    const int k_remainder = params->K - full_prefetch_iterations * PREFETCH_K;

    const short actual_bm =
        align_m ? BLOCK_ROWS
                : short(min(int(BLOCK_ROWS), int(params->M) - block_row_start));
    const short actual_bn =
        align_n ? BLOCK_COLS
                : short(min(int(BLOCK_COLS), int(params->N) - block_col_start));

    // Main loop
    for (int outer_k = 0; outer_k < full_prefetch_iterations; outer_k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (align_m && align_n) {
        loader_a.load_unsafe();
        loader_b.load_unsafe();
      } else {
        loader_a.load_safe(short2(PREFETCH_K, actual_bm));
        loader_b.load_safe(short2(PREFETCH_K, actual_bn));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      METAL_PRAGMA_UNROLL
      for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
        METAL_PRAGMA_UNROLL
        for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
          const short subtile_index = tile_m * TILES_N + tile_n;
          const short a_m_base = tile_row_offset + tile_m * SUBTILE_ROWS;
          const short b_n_base = tile_col_offset + tile_n * SUBTILE_COLS;

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < accumulator_capacity; i++) {
            accumulator_tensor[i] = accum_storage[subtile_index][i];
          }

          METAL_PRAGMA_UNROLL
          for (short k_step = 0; k_step < INNER_K_STEPS; k_step++) {
            const short k_offset = k_step * MATMUL_K_STEP;

            METAL_PRAGMA_UNROLL
            for (short i = 0; i < left_tensor.get_capacity(); i++) {
              auto coord = left_tensor.get_multidimensional_index(i);
              left_tensor[i] = left_shared
                  [(a_m_base + coord[1]) * THREADGROUP_LD + coord[0] +
                   k_offset];
            }

            METAL_PRAGMA_UNROLL
            for (short i = 0; i < right_tensor.get_capacity(); i++) {
              auto coord = right_tensor.get_multidimensional_index(i);
              right_tensor[i] = right_shared
                  [(b_n_base + coord[1]) * THREADGROUP_LD + coord[0] +
                   k_offset];
            }

            matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);
          }

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < accumulator_capacity; i++) {
            accum_storage[subtile_index][i] = accumulator_tensor[i];
          }
        }
      }

      loader_a.next();
      loader_b.next();
    }

    // Remainder loop
    if (k_remainder > 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_safe(short2(k_remainder, actual_bm));
      loader_b.load_safe(short2(k_remainder, actual_bn));
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const short remainder_steps =
          short((k_remainder + MATMUL_K_STEP - 1) / MATMUL_K_STEP);

      METAL_PRAGMA_UNROLL
      for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
        METAL_PRAGMA_UNROLL
        for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
          const short subtile_index = tile_m * TILES_N + tile_n;
          const short a_m_base = tile_row_offset + tile_m * SUBTILE_ROWS;
          const short b_n_base = tile_col_offset + tile_n * SUBTILE_COLS;

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < accumulator_capacity; i++) {
            accumulator_tensor[i] = accum_storage[subtile_index][i];
          }

          for (short k_step = 0; k_step < remainder_steps; k_step++) {
            const short k_offset = k_step * MATMUL_K_STEP;

            METAL_PRAGMA_UNROLL
            for (short i = 0; i < left_tensor.get_capacity(); i++) {
              auto coord = left_tensor.get_multidimensional_index(i);
              left_tensor[i] = left_shared
                  [(a_m_base + coord[1]) * THREADGROUP_LD + coord[0] +
                   k_offset];
            }

            METAL_PRAGMA_UNROLL
            for (short i = 0; i < right_tensor.get_capacity(); i++) {
              auto coord = right_tensor.get_multidimensional_index(i);
              right_tensor[i] = right_shared
                  [(b_n_base + coord[1]) * THREADGROUP_LD + coord[0] +
                   k_offset];
            }

            matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);
          }

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < accumulator_capacity; i++) {
            accum_storage[subtile_index][i] = accumulator_tensor[i];
          }
        }
      }
    }

    // Store results
    const int ld_d = params->leading_dimension_d;

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

        device T* out_ptr = output_ptr + row_offset * ld_d + col_offset;

        METAL_PRAGMA_UNROLL
        for (short i = 0; i < accumulator_capacity; i++) {
          auto coord = accumulator_tensor.get_multidimensional_index(i);
          const bool valid = accumulator_tensor.is_valid_element(i);
          AccumulatorType val = accum_storage[tile_m * TILES_N + tile_n][i];
          if (align_m && align_n) {
            if (valid) {
              out_ptr[coord[1] * ld_d + coord[0]] = T(val);
            }
          } else {
            if (valid && coord[1] < m_limit && coord[0] < n_limit) {
              out_ptr[coord[1] * ld_d + coord[0]] = T(val);
            }
          }
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu

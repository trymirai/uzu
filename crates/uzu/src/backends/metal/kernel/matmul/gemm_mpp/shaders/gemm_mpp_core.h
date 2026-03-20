#pragma once

#include "../../common/defines.h"
#include "../../../../generated/matmul.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

namespace uzu {
namespace matmul {

template <
    typename T,
    typename LoaderA,
    typename LoaderB,
    short BLOCK_M,
    short BLOCK_N,
    short PREFETCH_K,
    short THREADGROUP_LD,
    short SIMDGROUPS_M,
    short SIMDGROUPS_N,
    short SUBTILE_ROWS,
    short SUBTILE_COLS,
    short MATMUL_K_STEP,
    bool USE_NATIVE_LAYOUT = false>
METAL_FUNC void gemm_mpp_core(
    thread LoaderA& loader_a,
    thread LoaderB& loader_b,
    device T* output_matrix,
    const constant GemmParams* params,
    const bool align_m,
    const bool align_n,
    const int block_row_start,
    const int block_col_start,
    threadgroup T* left_shared,
    threadgroup T* right_shared,
    uint simd_group_id,
    uint simd_lane_id
) {
  using AccumulatorType = float;

  constexpr short SIMDGROUP_BLOCK_M = BLOCK_M / SIMDGROUPS_M;
  constexpr short SIMDGROUP_BLOCK_N = BLOCK_N / SIMDGROUPS_N;
  constexpr short TILES_M = SIMDGROUP_BLOCK_M / SUBTILE_ROWS;
  constexpr short TILES_N = SIMDGROUP_BLOCK_N / SUBTILE_COLS;
  constexpr short INNER_K_STEPS = PREFETCH_K / MATMUL_K_STEP;

  const short tile_row_offset =
      SIMDGROUP_BLOCK_M * (simd_group_id / SIMDGROUPS_N);
  const short tile_col_offset =
      SIMDGROUP_BLOCK_N * (simd_group_id % SIMDGROUPS_N);

  const size_t block_row_start_long = size_t(block_row_start);
  const size_t block_col_start_long = size_t(block_col_start);

  device T* output_ptr =
      output_matrix + block_row_start_long * params->leading_dimension_d +
      block_col_start_long + tile_row_offset * params->leading_dimension_d +
      tile_col_offset;

  const int simdgroup_limit_m_int =
      align_m ? int(SIMDGROUP_BLOCK_M)
              : min(int(SIMDGROUP_BLOCK_M),
                    params->M - (block_row_start + tile_row_offset));
  const short simdgroup_limit_m = short(simdgroup_limit_m_int);

  const int simdgroup_limit_n_int =
      align_n ? int(SIMDGROUP_BLOCK_N)
              : min(int(SIMDGROUP_BLOCK_N),
                    params->N - (block_col_start + tile_col_offset));
  const short simdgroup_limit_n = short(simdgroup_limit_n_int);

  const bool is_unaligned_m =
      align_m ? false : (simdgroup_limit_m != SIMDGROUP_BLOCK_M);
  const bool is_unaligned_n =
      align_n ? false : (simdgroup_limit_n != SIMDGROUP_BLOCK_N);

  constexpr auto matmul_descriptor = mpp::tensor_ops::matmul2d_descriptor(
      SUBTILE_ROWS,
      SUBTILE_COLS,
      MATMUL_K_STEP,
      false,
      true,
      USE_NATIVE_LAYOUT,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
  );

  mpp::tensor_ops::matmul2d<matmul_descriptor, metal::execution_simdgroup>
      matmul_operation;

  auto left_tensor =
      matmul_operation
          .template get_left_input_cooperative_tensor<T, T, AccumulatorType>();
  auto right_tensor =
      matmul_operation
          .template get_right_input_cooperative_tensor<T, T, AccumulatorType>();
  auto accumulator_tensor =
      matmul_operation.template get_destination_cooperative_tensor<
          decltype(left_tensor),
          decltype(right_tensor),
          AccumulatorType>();

  const short left_capacity = left_tensor.get_capacity();
  const short right_capacity = right_tensor.get_capacity();
  const short accumulator_capacity = accumulator_tensor.get_capacity();

  short left_col[16], left_row[16];
  short right_col[16], right_row[16];
  short output_col[16], output_row[16];
  bool output_valid[16];

  METAL_PRAGMA_UNROLL
  for (short i = 0; i < left_capacity; i++) {
    auto coord = left_tensor.get_multidimensional_index(i);
    left_col[i] = coord[0];
    left_row[i] = coord[1];
  }

  METAL_PRAGMA_UNROLL
  for (short i = 0; i < right_capacity; i++) {
    auto coord = right_tensor.get_multidimensional_index(i);
    right_col[i] = coord[0];
    right_row[i] = coord[1];
  }

  METAL_PRAGMA_UNROLL
  for (short i = 0; i < accumulator_capacity; i++) {
    auto coord = accumulator_tensor.get_multidimensional_index(i);
    output_col[i] = coord[0];
    output_row[i] = coord[1];
    output_valid[i] = accumulator_tensor.is_valid_element(i);
  }

  AccumulatorType accum_storage[TILES_M * TILES_N][16];
  int all_left_tg_base[TILES_M * TILES_N][16];
  int all_right_tg_base[TILES_M * TILES_N][16];

  METAL_PRAGMA_UNROLL
  for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
    METAL_PRAGMA_UNROLL
    for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
      const short subtile_index = tile_m * TILES_N + tile_n;
      const short a_m_base = tile_row_offset + tile_m * SUBTILE_ROWS;
      const short b_n_base = tile_col_offset + tile_n * SUBTILE_COLS;

      METAL_PRAGMA_UNROLL
      for (short i = 0; i < left_capacity; i++) {
        all_left_tg_base[subtile_index][i] =
            (a_m_base + left_row[i]) * THREADGROUP_LD + left_col[i];
      }

      METAL_PRAGMA_UNROLL
      for (short i = 0; i < right_capacity; i++) {
        all_right_tg_base[subtile_index][i] =
            (b_n_base + right_row[i]) * THREADGROUP_LD + right_col[i];
      }

      METAL_PRAGMA_UNROLL
      for (short i = 0; i < 16; i++) {
        accum_storage[subtile_index][i] = AccumulatorType(0);
      }
    }
  }

  const int full_prefetch_iterations = params->K / PREFETCH_K;
  const int k_remainder = params->K - full_prefetch_iterations * PREFETCH_K;

  const short actual_bm =
      align_m ? BLOCK_M : short(min(int(BLOCK_M), params->M - block_row_start));
  const short actual_bn =
      align_n ? BLOCK_N : short(min(int(BLOCK_N), params->N - block_col_start));

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

        const short m_limit =
            is_unaligned_m
                ? short(max(0, int(simdgroup_limit_m) - tile_m * SUBTILE_ROWS))
                : SUBTILE_ROWS;
        const short n_limit =
            is_unaligned_n
                ? short(max(0, int(simdgroup_limit_n) - tile_n * SUBTILE_COLS))
                : SUBTILE_COLS;
        if (m_limit <= 0 || n_limit <= 0) {
          continue;
        }

        METAL_PRAGMA_UNROLL
        for (short i = 0; i < accumulator_capacity; i++) {
          accumulator_tensor[i] = accum_storage[subtile_index][i];
        }

        METAL_PRAGMA_UNROLL
        for (short k_step = 0; k_step < INNER_K_STEPS; k_step++) {
          const short k_offset = k_step * MATMUL_K_STEP;

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < left_capacity; i++) {
            left_tensor[i] =
                left_shared[all_left_tg_base[subtile_index][i] + k_offset];
          }

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < right_capacity; i++) {
            right_tensor[i] =
                right_shared[all_right_tg_base[subtile_index][i] + k_offset];
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

        const short m_limit =
            is_unaligned_m
                ? short(max(0, int(simdgroup_limit_m) - tile_m * SUBTILE_ROWS))
                : SUBTILE_ROWS;
        const short n_limit =
            is_unaligned_n
                ? short(max(0, int(simdgroup_limit_n) - tile_n * SUBTILE_COLS))
                : SUBTILE_COLS;
        if (m_limit <= 0 || n_limit <= 0) {
          continue;
        }

        METAL_PRAGMA_UNROLL
        for (short i = 0; i < accumulator_capacity; i++) {
          accumulator_tensor[i] = accum_storage[subtile_index][i];
        }

        for (short k_step = 0; k_step < remainder_steps; k_step++) {
          const short k_offset = k_step * MATMUL_K_STEP;

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < left_capacity; i++) {
            left_tensor[i] =
                left_shared[all_left_tg_base[subtile_index][i] + k_offset];
          }

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < right_capacity; i++) {
            right_tensor[i] =
                right_shared[all_right_tg_base[subtile_index][i] + k_offset];
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

  METAL_PRAGMA_UNROLL
  for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
    METAL_PRAGMA_UNROLL
    for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
      const short row_offset = tile_m * SUBTILE_ROWS;
      const short col_offset = tile_n * SUBTILE_COLS;

      const short m_limit =
          is_unaligned_m ? short(max(0, int(simdgroup_limit_m) - row_offset))
                         : SUBTILE_ROWS;
      const short n_limit =
          is_unaligned_n ? short(max(0, int(simdgroup_limit_n) - col_offset))
                         : SUBTILE_COLS;
      if (m_limit <= 0 || n_limit <= 0) {
        continue;
      }

      device T* output_subtile_ptr =
          output_ptr + row_offset * params->leading_dimension_d + col_offset;

      METAL_PRAGMA_UNROLL
      for (short i = 0; i < accumulator_capacity; i++) {
        accumulator_tensor[i] = accum_storage[tile_m * TILES_N + tile_n][i];
      }

      const bool subtile_aligned_m =
          !is_unaligned_m || (m_limit == SUBTILE_ROWS);
      const bool subtile_aligned_n =
          !is_unaligned_n || (n_limit == SUBTILE_COLS);

      METAL_PRAGMA_UNROLL
      for (short i = 0; i < accumulator_capacity; i++) {
        if (subtile_aligned_m && subtile_aligned_n) {
          if (output_valid[i]) {
            output_subtile_ptr
                [output_row[i] * params->leading_dimension_d + output_col[i]] =
                    T(accumulator_tensor[i]);
          }
        } else {
          if (output_valid[i] && output_row[i] < m_limit &&
              output_col[i] < n_limit) {
            output_subtile_ptr
                [output_row[i] * params->leading_dimension_d + output_col[i]] =
                    T(accumulator_tensor[i]);
          }
        }
      }
    }
  }
}

} // namespace matmul
} // namespace uzu

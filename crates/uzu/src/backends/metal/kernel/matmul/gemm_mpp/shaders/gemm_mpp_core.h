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
    short MATMUL_K_STEP>
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
      align_m ? BLOCK_M
              : short(min(int(BLOCK_M), int(params->M) - block_row_start));
  const short actual_bn =
      align_n ? BLOCK_N
              : short(min(int(BLOCK_N), int(params->N) - block_col_start));

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
                [(a_m_base + coord[1]) * THREADGROUP_LD + coord[0] + k_offset];
          }

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < right_tensor.get_capacity(); i++) {
            auto coord = right_tensor.get_multidimensional_index(i);
            right_tensor[i] = right_shared
                [(b_n_base + coord[1]) * THREADGROUP_LD + coord[0] + k_offset];
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
                [(a_m_base + coord[1]) * THREADGROUP_LD + coord[0] + k_offset];
          }

          METAL_PRAGMA_UNROLL
          for (short i = 0; i < right_tensor.get_capacity(); i++) {
            auto coord = right_tensor.get_multidimensional_index(i);
            right_tensor[i] = right_shared
                [(b_n_base + coord[1]) * THREADGROUP_LD + coord[0] + k_offset];
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

} // namespace matmul
} // namespace uzu

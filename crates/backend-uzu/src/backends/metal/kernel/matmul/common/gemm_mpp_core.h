#pragma once

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

#include "defines.h"
#include "../../common/integral_constant.h"
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
    bool ALIGN_M,
    bool ALIGN_N,
    bool ALIGN_K,
    bool APPLY_AB_SCALE,
    bool IS_ACCUMULATE>
struct GemmMppCore {
  using AccumulatorType = float;
  METAL_CONST ushort BLOCK_K = 128;
  METAL_CONST ushort SIMDGROUP_COUNT =
      SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN;

  static METAL_FUNC void run(
      const device T* left_matrix,
      const device T* right_matrix,
      device T* output_matrix,
      const constant GemmParams* params,
      const float ab_scale,
      uint2 threadgroup_position
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

    const uint row_offset = tile_id_y * BLOCK_ROWS;
    const uint column_offset = tile_id_x * BLOCK_COLS;
    if (row_offset >= params->M || column_offset >= params->N) {
      return;
    }

    auto left_tensor_base = tensor(
        const_cast<device T*>(left_matrix),
        dextents<int, 2>{int(params->K), int(params->M)},
        array<int, 2>{1, int(params->leading_dimension_a)}
    );
    auto right_tensor_base = tensor(
        const_cast<device T*>(right_matrix),
        dextents<int, 2>{int(params->K), int(params->N)},
        array<int, 2>{1, int(params->leading_dimension_b)}
    );
    auto output_tensor_base = tensor(
        output_matrix,
        dextents<int, 2>{int(params->N), int(params->M)},
        array<int, 2>{1, int(params->leading_dimension_d)}
    );

    if constexpr (ALIGN_M && ALIGN_N) {
      auto left_tensor =
          left_tensor_base.template slice<dynamic_extent, BLOCK_ROWS>(
              0,
              int(row_offset)
          );
      auto right_tensor =
          right_tensor_base.template slice<dynamic_extent, BLOCK_COLS>(
              0,
              int(column_offset)
          );
      auto output_tensor =
          output_tensor_base.template slice<BLOCK_COLS, BLOCK_ROWS>(
              int(column_offset),
              int(row_offset)
          );
      multiply_and_store(left_tensor, right_tensor, output_tensor, params, ab_scale);
    } else {
      auto left_tensor = left_tensor_base.slice(0, int(row_offset));
      auto right_tensor = right_tensor_base.slice(0, int(column_offset));
      auto output_tensor =
          output_tensor_base.slice(int(column_offset), int(row_offset));
      multiply_and_store(left_tensor, right_tensor, output_tensor, params, ab_scale);
    }
  }

private:
  template <class LeftTensor, class RightTensor, class OutputTensor>
  static METAL_FUNC void multiply_and_store(
      thread LeftTensor& left_tensor,
      thread RightTensor& right_tensor,
      thread OutputTensor& output_tensor,
      const constant GemmParams* params,
      float ab_scale
  ) {
    if constexpr (ALIGN_K) {
      constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
          BLOCK_ROWS,
          BLOCK_COLS,
          BLOCK_K,
          false,
          true,
          false,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
      );
      mpp::tensor_ops::matmul2d<
          descriptor,
          execution_simdgroups<SIMDGROUP_COUNT>>
          matmul_operation;
      auto accumulator =
          matmul_operation.template get_destination_cooperative_tensor<
              LeftTensor,
              RightTensor,
              AccumulatorType>();

      METAL_PRAGMA_UNROLL
      for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
        if (accumulator.is_valid_element(index)) {
          accumulator[index] = AccumulatorType(0);
        }
      }

      const uint k_iterations = params->K / BLOCK_K;
      METAL_PRAGMA_NO_UNROLL
      for (uint k_index = 0; k_index < k_iterations; ++k_index) {
        threadgroup_barrier(mem_flags::mem_none);
        const int k_offset = int(k_index * BLOCK_K);
        auto left_tile =
            left_tensor.template slice<BLOCK_K, BLOCK_ROWS>(k_offset, 0);
        auto right_tile =
            right_tensor.template slice<BLOCK_K, BLOCK_COLS>(k_offset, 0);
        matmul_operation.run(left_tile, right_tile, accumulator);
      }

      store_accumulator(matmul_operation, left_tensor, right_tensor, output_tensor, accumulator, ab_scale);
    } else {
      constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
          BLOCK_ROWS,
          BLOCK_COLS,
          static_cast<int>(dynamic_extent),
          false,
          true,
          false
      );
      mpp::tensor_ops::matmul2d<
          descriptor,
          execution_simdgroups<SIMDGROUP_COUNT>>
          matmul_operation;
      auto accumulator =
          matmul_operation.template get_destination_cooperative_tensor<
              LeftTensor,
              RightTensor,
              AccumulatorType>();

      METAL_PRAGMA_UNROLL
      for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
        if (accumulator.is_valid_element(index)) {
          accumulator[index] = AccumulatorType(0);
        }
      }

      matmul_operation.run(left_tensor, right_tensor, accumulator);
      store_accumulator(matmul_operation, left_tensor, right_tensor, output_tensor, accumulator, ab_scale);
    }
  }

  template <
      class MatmulOperation,
      class LeftTensor,
      class RightTensor,
      class OutputTensor,
      class Accumulator>
  static METAL_FUNC void store_accumulator(
      thread MatmulOperation& matmul_operation,
      thread LeftTensor& left_tensor,
      thread RightTensor& right_tensor,
      thread OutputTensor& output_tensor,
      thread Accumulator& accumulator,
      float ab_scale
  ) {
    auto output_values =
        matmul_operation.template get_destination_cooperative_tensor<
            LeftTensor,
            RightTensor,
            T>();

    if constexpr (IS_ACCUMULATE) {
      output_values.load(output_tensor);
    }

    METAL_PRAGMA_UNROLL
    for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
      if (!accumulator.is_valid_element(index)) {
        continue;
      }

      float value = accumulator[index];
      if constexpr (APPLY_AB_SCALE) {
        value *= ab_scale;
      }
      if constexpr (IS_ACCUMULATE) {
        value += float(output_values[index]);
      }
      output_values[index] = T(value);
    }

    output_values.store(output_tensor);
  }
};

} // namespace matmul
} // namespace uzu

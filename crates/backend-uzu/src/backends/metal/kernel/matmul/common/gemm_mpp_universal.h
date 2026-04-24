#pragma once

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

#include "defines.h"
#include "../../common/integral_constant.h"
#include "../../generated/matmul.h"

using namespace metal;

namespace uzu {
namespace matmul {

template <typename T, bool APPLY_AB_SCALE, bool IS_ACCUMULATE>
struct GemmMppUniversal {
  using AccumulatorType = float;

  METAL_CONST ushort BLOCK_ROWS = 32;
  METAL_CONST ushort BLOCK_COLS = 32;

  static METAL_FUNC void run(
      const device T*       left_matrix,
      const device T*       right_matrix,
      device T*             output_matrix,
      const constant GemmParams* params,
      float                 ab_scale,
      uint2                 threadgroup_position) {
    const uint tile_column = threadgroup_position.x;
    const uint tile_row = threadgroup_position.y;
    if (tile_column >= params->threadgroups_per_row ||
        tile_row >= params->threadgroups_per_column) {
      return;
    }

    const uint row_offset = tile_row * BLOCK_ROWS;
    const uint column_offset = tile_column * BLOCK_COLS;
    if (row_offset >= params->M || column_offset >= params->N) {
      return;
    }

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        BLOCK_ROWS,
        BLOCK_COLS,
        static_cast<int>(dynamic_extent),
        /*transpose_left=*/false,
        /*transpose_right=*/true,
        /*relaxed_precision=*/false);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> matmul_operation;

    auto left_tensor_base = tensor(
        const_cast<device T*>(left_matrix),
        dextents<int, 2>{int(params->K), int(params->M)},
        array<int, 2>{1, int(params->leading_dimension_a)});
    auto right_tensor_base = tensor(
        const_cast<device T*>(right_matrix),
        dextents<int, 2>{int(params->K), int(params->N)},
        array<int, 2>{1, int(params->leading_dimension_b)});

    dispatch_bool(
        row_offset + BLOCK_ROWS <= params->M &&
            column_offset + BLOCK_COLS <= params->N,
        [&](auto full_tile) {
          if constexpr (full_tile.value) {
            auto left_tensor =
                left_tensor_base.template slice<dynamic_extent, BLOCK_ROWS>(
                    /*offset_x=*/0,
                    int(row_offset));
            auto right_tensor =
                right_tensor_base.template slice<dynamic_extent, BLOCK_COLS>(
                    /*offset_x=*/0,
                    int(column_offset));
            accumulate_and_store(
                matmul_operation,
                left_tensor,
                right_tensor,
                output_matrix,
                params,
                row_offset,
                column_offset,
                ab_scale);
          } else {
            auto left_tensor =
                left_tensor_base.slice(/*offset_x=*/0, int(row_offset));
            auto right_tensor =
                right_tensor_base.slice(/*offset_x=*/0, int(column_offset));
            accumulate_and_store(
                matmul_operation,
                left_tensor,
                right_tensor,
                output_matrix,
                params,
                row_offset,
                column_offset,
                ab_scale);
          }
        });
  }

private:
  template <class MatmulOperation, class LeftTensor, class RightTensor>
  static METAL_FUNC void accumulate_and_store(
      const thread MatmulOperation& matmul_operation,
      thread LeftTensor&            left_tensor,
      thread RightTensor&           right_tensor,
      device T*                     output_matrix,
      const constant GemmParams*    params,
      uint                          row_offset,
      uint                          column_offset,
      float                         ab_scale) {
    auto accumulator = matmul_operation
        .template get_destination_cooperative_tensor<
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

    METAL_PRAGMA_UNROLL
    for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
      if (!accumulator.is_valid_element(index)) {
        continue;
      }

      const auto local_coordinates =
          accumulator.get_multidimensional_index(index);
      const uint output_column = column_offset + uint(local_coordinates[0]);
      const uint output_row = row_offset + uint(local_coordinates[1]);
      if (output_row >= params->M || output_column >= params->N) {
        continue;
      }

      float value = accumulator[index];
      if constexpr (APPLY_AB_SCALE) {
        value *= ab_scale;
      }
      if constexpr (IS_ACCUMULATE) {
        value += float(output_matrix[output_row * params->leading_dimension_d +
                                     output_column]);
      }
      output_matrix[output_row * params->leading_dimension_d + output_column] =
          T(value);
    }
  }
};

} // namespace matmul
} // namespace uzu

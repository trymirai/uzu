#pragma once

#include "simdgroup_fragment_layout.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

namespace uzu {
namespace matmul {

/// Performs a complete GEMM for one output subtile using Metal's cooperative
/// tensor API.
///
/// Loads A and B directly into cooperative tensors using
/// get_multidimensional_index() for correct coordinate mapping on all Apple
/// Silicon generations (M1-M4). Coordinates are precomputed once before the K
/// loop for efficiency.
///
/// Template parameters:
///   subtile_rows, subtile_columns — output subtile dimensions (e.g. 16×32)
///   unroll_k — K dimension per matmul2d invocation (typically 16)
///   AccumulatorType — accumulator precision (float or int32)
///   LeftInputType, RightInputType — input element types
///   OutputType — output element type
///   transpose_left, transpose_right — transpose flags for A and B
template <
    short subtile_rows,
    short subtile_columns,
    short unroll_k,
    typename AccumulatorType,
    typename LeftInputType,
    typename RightInputType,
    typename OutputType,
    bool transpose_left,
    bool transpose_right,
    bool aligned_rows = false,
    bool aligned_columns = false,
    bool aligned_k = false
>
METAL_FUNC void cooperative_tensor_gemm(
    const device LeftInputType* left_input_pointer,
    int leading_dimension_a,
    const device RightInputType* right_input_pointer,
    int leading_dimension_b,
    device OutputType* output_pointer,
    int leading_dimension_output,
    int k_dimension,
    short row_limit,
    short column_limit
) {

  constexpr auto matmul_descriptor = mpp::tensor_ops::matmul2d_descriptor(
      subtile_rows,
      subtile_columns,
      unroll_k,
      transpose_left,
      transpose_right,
      false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
  );

  mpp::tensor_ops::matmul2d<matmul_descriptor, metal::execution_simdgroup>
      matmul_operation;

  auto left_tensor =
      matmul_operation.template get_left_input_cooperative_tensor<
          LeftInputType,
          RightInputType,
          AccumulatorType
      >();
  auto right_tensor =
      matmul_operation.template get_right_input_cooperative_tensor<
          LeftInputType,
          RightInputType,
          AccumulatorType
      >();
  auto accumulator_tensor =
      matmul_operation.template get_destination_cooperative_tensor<
          decltype(left_tensor),
          decltype(right_tensor),
          AccumulatorType
      >();

#ifdef MPP_LAYOUT_PROBE
  if (k_dimension == -999) {
    device int* probe = reinterpret_cast<device int*>(output_pointer);
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    if (lane == 0) {
      probe[0] = left_tensor.get_capacity();
      probe[1] = right_tensor.get_capacity();
      probe[2] = accumulator_tensor.get_capacity();
    }
    const int left_capacity = left_tensor.get_capacity();
    const int right_capacity = right_tensor.get_capacity();
    const int accumulator_capacity = accumulator_tensor.get_capacity();
    const int per_lane =
        (left_capacity + right_capacity + accumulator_capacity) * 2 + 8 * 2;
    const int base = 3 + lane * per_lane;
    int offset = base;
    for (int i = 0; i < left_capacity; i++) {
      auto coord = left_tensor.get_multidimensional_index(i);
      probe[offset++] = coord[0];
      probe[offset++] = coord[1];
    }
    for (int i = 0; i < right_capacity; i++) {
      auto coord = right_tensor.get_multidimensional_index(i);
      probe[offset++] = coord[0];
      probe[offset++] = coord[1];
    }
    for (int i = 0; i < accumulator_capacity; i++) {
      auto coord = accumulator_tensor.get_multidimensional_index(i);
      probe[offset++] = coord[0];
      probe[offset++] = coord[1];
    }
    for (short j = 0; j < 8; j++) {
      short2 fragment_coord = SimdgroupFragmentLayout::get_coordinate(j);
      probe[offset++] = fragment_coord.x;
      probe[offset++] = fragment_coord.y;
    }
    return;
  }
#endif

  const short left_capacity = left_tensor.get_capacity();
  const short right_capacity = right_tensor.get_capacity();
  const short accumulator_capacity = accumulator_tensor.get_capacity();

  short left_col[16], left_row[16];
  short right_col[16], right_row[16];
  short output_col[16], output_row[16];
  bool output_valid[16];

  PRAGMA_UNROLL
  for (short i = 0; i < left_capacity; i++) {
    auto coord = left_tensor.get_multidimensional_index(i);
    left_col[i] = coord[0];
    left_row[i] = coord[1];
  }

  PRAGMA_UNROLL
  for (short i = 0; i < right_capacity; i++) {
    auto coord = right_tensor.get_multidimensional_index(i);
    right_col[i] = coord[0];
    right_row[i] = coord[1];
  }

  PRAGMA_UNROLL
  for (short i = 0; i < accumulator_capacity; i++) {
    auto coord = accumulator_tensor.get_multidimensional_index(i);
    output_col[i] = coord[0];
    output_row[i] = coord[1];
    output_valid[i] = accumulator_tensor.is_valid_element(i);
  }

  int left_load_offset[16];
  int right_load_offset[16];
  bool left_in_bounds[16];
  bool right_in_bounds[16];

  PRAGMA_UNROLL
  for (short i = 0; i < left_capacity; i++) {
    if constexpr (!transpose_left) {
      left_load_offset[i] = left_row[i] * leading_dimension_a + left_col[i];
      left_in_bounds[i] = aligned_rows || (left_row[i] < row_limit);
    } else {
      left_load_offset[i] = left_col[i] * leading_dimension_a + left_row[i];
      left_in_bounds[i] = aligned_rows || (left_col[i] < row_limit);
    }
  }

  PRAGMA_UNROLL
  for (short i = 0; i < right_capacity; i++) {
    if constexpr (!transpose_right) {
      right_load_offset[i] = right_row[i] * leading_dimension_b + right_col[i];
      right_in_bounds[i] = aligned_columns || (right_col[i] < column_limit);
    } else {
      right_load_offset[i] = right_row[i] * leading_dimension_b + right_col[i];
      right_in_bounds[i] = aligned_columns || (right_row[i] < column_limit);
    }
  }

  PRAGMA_UNROLL
  for (short i = 0; i < accumulator_capacity; i++) {
    accumulator_tensor[i] = AccumulatorType(0);
  }

  const int aligned_k_iterations =
      aligned_k ? k_dimension : (k_dimension / unroll_k) * unroll_k;

  for (int k = 0; k < aligned_k_iterations; k += unroll_k) {
    PRAGMA_UNROLL
    for (short i = 0; i < left_capacity; i++) {
      if constexpr (aligned_rows) {
        left_tensor[i] = left_input_pointer[left_load_offset[i]];
      } else {
        left_tensor[i] = left_in_bounds[i]
                             ? left_input_pointer[left_load_offset[i]]
                             : LeftInputType(0);
      }
    }

    PRAGMA_UNROLL
    for (short i = 0; i < right_capacity; i++) {
      if constexpr (aligned_columns) {
        right_tensor[i] = right_input_pointer[right_load_offset[i]];
      } else {
        right_tensor[i] = right_in_bounds[i]
                              ? right_input_pointer[right_load_offset[i]]
                              : RightInputType(0);
      }
    }

    matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);

    left_input_pointer +=
        transpose_left ? (unroll_k * leading_dimension_a) : unroll_k;
    right_input_pointer +=
        transpose_right ? unroll_k : (unroll_k * leading_dimension_b);
  }

  if constexpr (!aligned_k) {
    if (aligned_k_iterations < k_dimension) {
      const short k_remaining = short(k_dimension - aligned_k_iterations);

      PRAGMA_UNROLL
      for (short i = 0; i < left_capacity; i++) {
        const short k_coord = transpose_left ? left_row[i] : left_col[i];
        if (left_in_bounds[i] && k_coord < k_remaining) {
          left_tensor[i] = left_input_pointer[left_load_offset[i]];
        } else {
          left_tensor[i] = LeftInputType(0);
        }
      }

      PRAGMA_UNROLL
      for (short i = 0; i < right_capacity; i++) {
        const short k_coord = transpose_right ? right_col[i] : right_row[i];
        if (right_in_bounds[i] && k_coord < k_remaining) {
          right_tensor[i] = right_input_pointer[right_load_offset[i]];
        } else {
          right_tensor[i] = RightInputType(0);
        }
      }

      matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);
    }
  }

  PRAGMA_UNROLL
  for (short i = 0; i < accumulator_capacity; i++) {
    if constexpr (aligned_rows && aligned_columns) {
      if (output_valid[i]) {
        output_pointer
            [output_row[i] * leading_dimension_output + output_col[i]] =
                OutputType(accumulator_tensor[i]);
      }
    } else {
      if (output_valid[i] && output_row[i] < row_limit &&
          output_col[i] < column_limit) {
        output_pointer
            [output_row[i] * leading_dimension_output + output_col[i]] =
                OutputType(accumulator_tensor[i]);
      }
    }
  }
}

} // namespace matmul
} // namespace uzu

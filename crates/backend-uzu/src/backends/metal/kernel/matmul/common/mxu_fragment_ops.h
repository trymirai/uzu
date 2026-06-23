#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
using namespace uzu;

#include "defines.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace uzu {
namespace matmul {

struct MxuFragmentOps {
  METAL_CONST ushort FRAGMENT_ROWS = 16;
  METAL_CONST ushort FRAGMENT_COLS = 16;

  METAL_CONST ushort ELEMENTS_PER_THREAD = (FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE;

  METAL_CONST ushort THREAD_ELEMENT_ROWS = 2;
  METAL_CONST ushort THREAD_ELEMENT_COLS = 4;

  METAL_CONST ushort THREAD_ELEMENT_ROW_STRIDE = 8;

  static_assert(
      THREAD_ELEMENT_ROWS * THREAD_ELEMENT_COLS == ELEMENTS_PER_THREAD,
      "MxuFragment shape is not consistent with element count"
  );

  template <typename U>
  using ThreadVector = typename metal::vec<U, ELEMENTS_PER_THREAD>;

  // MPP has no valid 16x16x16 op; fragment_matmul pairs fragments into 16x32.
  template <typename CType, typename AType, typename BType, bool transpose_a, bool transpose_b, typename MarshalInputs>
  METAL_FUNC static void mma_impl(
      thread ThreadVector<CType>& output_0,
      thread ThreadVector<CType>& output_1,
      MarshalInputs marshal_inputs
  ) {
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        FRAGMENT_ROWS,
        2 * FRAGMENT_COLS,
        FRAGMENT_COLS,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );

    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

    auto cooperative_left = matmul_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_right = matmul_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_output = matmul_op.template get_destination_cooperative_tensor<
        decltype(cooperative_left),
        decltype(cooperative_right),
        CType>();

    marshal_inputs(cooperative_left, cooperative_right);

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      cooperative_output[i] = output_0[i];
      cooperative_output[ELEMENTS_PER_THREAD + i] = output_1[i];
    }

    matmul_op.run(cooperative_left, cooperative_right, cooperative_output);

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      output_0[i] = cooperative_output[i];
      output_1[i] = cooperative_output[ELEMENTS_PER_THREAD + i];
    }
  }

  template <typename CType, typename AType, typename BType, bool transpose_a = false, bool transpose_b = false>
  METAL_FUNC static void mma(
      thread ThreadVector<CType>& output_col_0,
      thread ThreadVector<CType>& output_col_1,
      const thread ThreadVector<AType>& left,
      metal::bool_constant<transpose_a>,
      const thread ThreadVector<BType>& right_col_0,
      const thread ThreadVector<BType>& right_col_1,
      metal::bool_constant<transpose_b>
  ) {
    mma_impl<CType, AType, BType, transpose_a, transpose_b>(
        output_col_0,
        output_col_1,
        [&](thread auto& cooperative_left, thread auto& cooperative_right) {
          METAL_PRAGMA_UNROLL
          for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
            cooperative_left[i] = left[i];
            cooperative_right[i] = right_col_0[i];
            cooperative_right[ELEMENTS_PER_THREAD + i] = right_col_1[i];
          }
        }
    );
  }

  template <typename CType, typename AType, typename BType, bool transpose_a = false, bool transpose_b = false>
  METAL_FUNC static void mma(
      thread ThreadVector<CType>& output_row_0,
      thread ThreadVector<CType>& output_row_1,
      const thread ThreadVector<AType>& left_row_0,
      const thread ThreadVector<AType>& left_row_1,
      metal::bool_constant<transpose_a>,
      const thread ThreadVector<BType>& right,
      metal::bool_constant<transpose_b>
  ) {
    mma_impl<CType, AType, BType, transpose_a, transpose_b>(
        output_row_0,
        output_row_1,
        [&](thread auto& cooperative_left, thread auto& cooperative_right) {
          METAL_PRAGMA_UNROLL
          for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
            cooperative_left[i] = left_row_0[i];
            cooperative_left[ELEMENTS_PER_THREAD + i] = left_row_1[i];
            cooperative_right[i] = right[i];
          }
        }
    );
  }

  template <bool transpose_a, bool transpose_b, class OutputFragment, class LeftFragment, class RightFragment>
  METAL_FUNC static void fragment_matmul(
      thread OutputFragment& output,
      thread LeftFragment& left,
      thread RightFragment& right
  ) {
    constexpr ushort left_rows = transpose_a ? LeftFragment::COL_FRAGMENTS : LeftFragment::ROW_FRAGMENTS;
    constexpr ushort rows = OutputFragment::ROW_FRAGMENTS;
    static_assert(left_rows == rows, "fragment matmul: M dimensions do not match");

    constexpr ushort right_cols = transpose_b ? RightFragment::ROW_FRAGMENTS : RightFragment::COL_FRAGMENTS;
    constexpr ushort cols = OutputFragment::COL_FRAGMENTS;
    static_assert(right_cols == cols, "fragment matmul: N dimensions do not match");

    constexpr ushort left_depth = transpose_a ? LeftFragment::ROW_FRAGMENTS : LeftFragment::COL_FRAGMENTS;
    constexpr ushort depth = transpose_b ? RightFragment::COL_FRAGMENTS : RightFragment::ROW_FRAGMENTS;
    static_assert(left_depth == depth, "fragment matmul: K dimensions do not match");

    // Other shapes silently emit no MMAs.
    static_assert(
        (cols % 2 == 0) || (cols == 1 && rows % 2 == 0),
        "MXU fragment_matmul requires even N, or N==1 with even M (MPP pairing)"
    );

    constexpr auto transpose_left = metal::bool_constant<transpose_a>{};
    constexpr auto transpose_right = metal::bool_constant<transpose_b>{};

    if constexpr (cols == 1 && rows % 2 == 0) {
      METAL_PRAGMA_UNROLL
      for (ushort row = 0; row < rows; row += 2) {
        METAL_PRAGMA_UNROLL
        for (ushort col = 0; col < cols; ++col) {
          METAL_PRAGMA_UNROLL
          for (ushort k = 0; k < depth; ++k) {
            mma<typename OutputFragment::ElementType,
                typename LeftFragment::ElementType,
                typename RightFragment::ElementType,
                transpose_a,
                transpose_b>(
                output.fragment_at(row, col),
                output.fragment_at(row + 1, col),
                left.fragment_at(row, k, transpose_left),
                left.fragment_at(row + 1, k, transpose_left),
                transpose_left,
                right.fragment_at(k, col, transpose_right),
                transpose_right
            );
          }
        }
      }
    } else if constexpr (cols % 2 == 0) {
      METAL_PRAGMA_UNROLL
      for (ushort row = 0; row < rows; ++row) {
        METAL_PRAGMA_UNROLL
        for (ushort col = 0; col < cols; col += 2) {
          METAL_PRAGMA_UNROLL
          for (ushort k = 0; k < depth; ++k) {
            mma<typename OutputFragment::ElementType,
                typename LeftFragment::ElementType,
                typename RightFragment::ElementType,
                transpose_a,
                transpose_b>(
                output.fragment_at(row, col),
                output.fragment_at(row, col + 1),
                left.fragment_at(row, k, transpose_left),
                transpose_left,
                right.fragment_at(k, col, transpose_right),
                right.fragment_at(k, col + 1, transpose_right),
                transpose_right
            );
          }
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu

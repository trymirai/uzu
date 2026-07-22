#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
using namespace uzu;

#include "defines.h"
#include "loader.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace uzu {
namespace matmul {

UZU_CONST ushort MXU_MMA_ROWS = 16;
UZU_CONST ushort MXU_MMA_COLS = 16;

// RELAXED=false uses the strict MPP layout; it currently performs about the same as simdgroup.
template <bool RELAXED = true>
struct MxuFragmentOps {
  UZU_CONST ushort FRAGMENT_ROWS = MXU_MMA_ROWS;
  UZU_CONST ushort FRAGMENT_COLS = MXU_MMA_COLS;
  UZU_CONST bool READ_TRANSPOSE_SWAPS_SOURCE_STRIDES = false;
  using BlockStorage = DeviceBlockStorage;

  UZU_CONST ushort ELEMENTS_PER_THREAD = (FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE;

  UZU_CONST ushort THREAD_ELEMENT_ROWS = 2;
  UZU_CONST ushort THREAD_ELEMENT_COLS = 4;

  UZU_CONST ushort THREAD_ELEMENT_ROW_STRIDE = FRAGMENT_ROWS / THREAD_ELEMENT_ROWS;

  static_assert(
      THREAD_ELEMENT_ROWS * THREAD_ELEMENT_COLS == ELEMENTS_PER_THREAD,
      "MxuFragment shape is not consistent with element count"
  );

  template <typename U>
  using ThreadVector = typename metal::vec<U, ELEMENTS_PER_THREAD>;

  METAL_FUNC static constexpr short2 get_position(ushort simd_lane_id) {
    if constexpr (RELAXED) {
      const short quad = simd_lane_id / 4;
      const short row = (quad & 4) + (simd_lane_id / 2) % 4;
      const short col = ((quad & 2) + simd_lane_id % 2) * THREAD_ELEMENT_COLS;
      return short2{col, row};
    } else {
      const short col = short((simd_lane_id & 1) * 2 + ((simd_lane_id >> 3) & 1) * 4);
      const short row = short(((simd_lane_id >> 1) & 3) + ((simd_lane_id >> 4) & 1) * 4);
      return short2{col, row};
    }
  }

  METAL_FUNC static constexpr short2 get_element_offset(ushort element_index) {
    if constexpr (RELAXED) {
      const short row = short((element_index / THREAD_ELEMENT_COLS) * THREAD_ELEMENT_ROW_STRIDE);
      const short col = short(element_index % THREAD_ELEMENT_COLS);
      return short2{col, row};
    } else {
      const short row = short((element_index / 4) * 8);
      const ushort col_slot = element_index & 3;
      const short col = short((col_slot & 1) + (col_slot / 2) * 8);
      return short2{col, row};
    }
  }

  // MPP has no valid 16x16x16 op; fragment_mma pairs fragments into 16x32.
  template <
      bool ACCUMULATE,
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a,
      bool transpose_b,
      typename MarshalInputs>
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
        RELAXED,
        ACCUMULATE ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
                   : mpp::tensor_ops::matmul2d_descriptor::mode::multiply
    );

    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

    auto cooperative_left = matmul_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_right = matmul_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_output = matmul_op.template get_destination_cooperative_tensor<
        decltype(cooperative_left),
        decltype(cooperative_right),
        CType>();

    marshal_inputs(cooperative_left, cooperative_right);

    if constexpr (ACCUMULATE) {
      if constexpr (RELAXED) {
        METAL_PRAGMA_UNROLL
        for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
          cooperative_output[i] = output_0[i];
          cooperative_output[ELEMENTS_PER_THREAD + i] = output_1[i];
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (ushort i = 0; i < 4; i++) {
          cooperative_output[i] = output_0[i];
          cooperative_output[4 + i] = output_1[i];
          cooperative_output[8 + i] = output_0[4 + i];
          cooperative_output[12 + i] = output_1[4 + i];
        }
      }
    }

    matmul_op.run(cooperative_left, cooperative_right, cooperative_output);

    if constexpr (RELAXED) {
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
        output_0[i] = cooperative_output[i];
        output_1[i] = cooperative_output[ELEMENTS_PER_THREAD + i];
      }
    } else {
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < 4; i++) {
        output_0[i] = cooperative_output[i];
        output_1[i] = cooperative_output[4 + i];
        output_0[4 + i] = cooperative_output[8 + i];
        output_1[4 + i] = cooperative_output[12 + i];
      }
    }
  }

  template <
      bool ACCUMULATE,
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static void matmul(
      thread ThreadVector<CType>& output_col_0,
      thread ThreadVector<CType>& output_col_1,
      const thread ThreadVector<AType>& left,
      metal::bool_constant<transpose_a>,
      const thread ThreadVector<BType>& right_col_0,
      const thread ThreadVector<BType>& right_col_1,
      metal::bool_constant<transpose_b>
  ) {
    mma_impl<ACCUMULATE, CType, AType, BType, transpose_a, transpose_b>(
        output_col_0,
        output_col_1,
        [&](thread auto& cooperative_left, thread auto& cooperative_right) {
          METAL_PRAGMA_UNROLL
          for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
            cooperative_left[i] = left[i];
          }
          if constexpr (RELAXED) {
            METAL_PRAGMA_UNROLL
            for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
              cooperative_right[i] = right_col_0[i];
              cooperative_right[ELEMENTS_PER_THREAD + i] = right_col_1[i];
            }
          } else {
            METAL_PRAGMA_UNROLL
            for (ushort i = 0; i < 4; i++) {
              cooperative_right[i] = right_col_0[i];
              cooperative_right[4 + i] = right_col_1[i];
              cooperative_right[8 + i] = right_col_0[4 + i];
              cooperative_right[12 + i] = right_col_1[4 + i];
            }
          }
        }
    );
  }

  template <
      bool ACCUMULATE,
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static void matmul(
      thread ThreadVector<CType>& output_row_0,
      thread ThreadVector<CType>& output_row_1,
      const thread ThreadVector<AType>& left_row_0,
      const thread ThreadVector<AType>& left_row_1,
      metal::bool_constant<transpose_a>,
      const thread ThreadVector<BType>& right,
      metal::bool_constant<transpose_b>
  ) {
    static_assert(RELAXED, "strict MXU row-pairing is not implemented");
    mma_impl<ACCUMULATE, CType, AType, BType, transpose_a, transpose_b>(
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

  template <
      bool ACCUMULATE,
      bool transpose_a,
      bool transpose_b,
      class OutputFragment,
      class LeftFragment,
      class RightFragment>
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

    static_assert(
        (cols % 2 == 0) || (cols == 1 && rows % 2 == 0),
        "MXU fragment_mma requires even N, or N==1 with even M (MPP pairing)"
    );

    constexpr auto transpose_left = metal::bool_constant<transpose_a>{};
    constexpr auto transpose_right = metal::bool_constant<transpose_b>{};

    if constexpr (cols == 1 && rows % 2 == 0) {
      METAL_PRAGMA_UNROLL
      for (ushort row = 0; row < rows; row += 2) {
        METAL_PRAGMA_UNROLL
        for (ushort col = 0; col < cols; ++col) {
          if constexpr (!ACCUMULATE) {
            matmul<
                false,
                typename OutputFragment::ElementType,
                typename LeftFragment::ElementType,
                typename RightFragment::ElementType,
                transpose_a,
                transpose_b>(
                output.fragment_at(row, col),
                output.fragment_at(row + 1, col),
                left.fragment_at(row, 0, transpose_left),
                left.fragment_at(row + 1, 0, transpose_left),
                transpose_left,
                right.fragment_at(0, col, transpose_right),
                transpose_right
            );
          }
          METAL_PRAGMA_UNROLL
          for (ushort k = ACCUMULATE ? 0 : 1; k < depth; ++k) {
            matmul<
                true,
                typename OutputFragment::ElementType,
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
          if constexpr (!ACCUMULATE) {
            matmul<
                false,
                typename OutputFragment::ElementType,
                typename LeftFragment::ElementType,
                typename RightFragment::ElementType,
                transpose_a,
                transpose_b>(
                output.fragment_at(row, col),
                output.fragment_at(row, col + 1),
                left.fragment_at(row, 0, transpose_left),
                transpose_left,
                right.fragment_at(0, col, transpose_right),
                right.fragment_at(0, col + 1, transpose_right),
                transpose_right
            );
          }
          METAL_PRAGMA_UNROLL
          for (ushort k = ACCUMULATE ? 0 : 1; k < depth; ++k) {
            matmul<
                true,
                typename OutputFragment::ElementType,
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

  template <bool transpose_a, bool transpose_b, class OutputFragment, class LeftFragment, class RightFragment>
  METAL_FUNC static void fragment_mma(
      thread OutputFragment& output,
      thread LeftFragment& left,
      thread RightFragment& right
  ) {
    fragment_matmul<true, transpose_a, transpose_b>(output, left, right);
  }

  template <bool transpose_a, bool transpose_b, class OutputFragment, class LeftFragment, class RightFragment>
  METAL_FUNC static void fragment_mm(
      thread OutputFragment& output,
      thread LeftFragment& left,
      thread RightFragment& right
  ) {
    // MXU relaxed multiply is slightly faster than multiply_accumulate for pure matmul.
    fragment_matmul<false, transpose_a, transpose_b>(output, left, right);
  }
};

using MxuStrictFragmentOps = MxuFragmentOps<false>;

} // namespace matmul
} // namespace uzu

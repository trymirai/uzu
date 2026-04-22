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

  METAL_CONST ushort ELEMENTS_PER_THREAD =
      (FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE;

  METAL_CONST ushort THREAD_ELEMENT_ROWS = 2;
  METAL_CONST ushort THREAD_ELEMENT_COLS = 4;

  METAL_CONST ushort THREAD_ELEMENT_ROW_STRIDE = 8;

  static_assert(
      THREAD_ELEMENT_ROWS * THREAD_ELEMENT_COLS == ELEMENTS_PER_THREAD,
      "MxuFragment shape is not consistent with element count"
  );

  template <typename U>
  using ThreadVector = typename metal::vec<U, ELEMENTS_PER_THREAD>;

  METAL_FUNC static short2 get_position(
      const thread ThreadContext& thread_context
  ) {
    const ushort simdgroup_index = ushort(thread_context.simdgroup_index);
    const short quad_id = simdgroup_index / 4;
    const short thread_row = (quad_id & 4) + (simdgroup_index / 2) % 4;
    const short thread_col =
        ((quad_id & 2) + simdgroup_index % 2) * THREAD_ELEMENT_COLS;
    return short2{thread_col, thread_row};
  }

  template <
      typename T,
      typename SourcePointerType,
      typename RowStride,
      typename ColStride,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void load(
      thread ThreadVector<T>& destination,
      SourcePointerType source,
      RowStride row_stride,
      ColStride col_stride,
      RowOffset row_offset,
      ColOffset col_offset,
      const thread ThreadContext& thread_context
  ) {
    const short2 position = get_position(thread_context);
    source += position.y * row_stride + position.x * col_stride;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      const auto row = row_offset + i * THREAD_ELEMENT_ROW_STRIDE;
      const auto col = col_offset;

      if constexpr (metal::is_same_v<ColStride, Int<1>>) {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
          destination[i * THREAD_ELEMENT_COLS + j] =
              static_cast<T>(source[row * row_stride + col + j]);
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
          destination[i * THREAD_ELEMENT_COLS + j] =
              static_cast<T>(source[row * row_stride + (col + j) * col_stride]);
        }
      }
    }
  }

  template <
      typename T,
      typename SourcePointerType,
      typename RowStride,
      typename ColStride,
      typename RowLimit,
      typename ColLimit,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread ThreadVector<T>& destination,
      SourcePointerType source,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      RowOffset row_offset,
      ColOffset col_offset,
      const thread ThreadContext& thread_context
  ) {
    const short2 position = get_position(thread_context);
    source += position.y * row_stride + position.x * col_stride;
    auto local_row_limit = row_limit - position.y;
    auto local_col_limit = col_limit - position.x;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      const auto row = row_offset + i * THREAD_ELEMENT_ROW_STRIDE;
      const auto col = col_offset;
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
        if ((row < local_row_limit) && ((col + j) < local_col_limit)) {
          destination[i * THREAD_ELEMENT_COLS + j] =
              static_cast<T>(source[row * row_stride + (col + j) * col_stride]);
        } else {
          destination[i * THREAD_ELEMENT_COLS + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DestinationPointerType,
      typename RowStride,
      typename ColStride,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread ThreadVector<T>& source,
      DestinationPointerType destination,
      RowStride row_stride,
      ColStride col_stride,
      RowOffset row_offset,
      ColOffset col_offset,
      const thread ThreadContext& thread_context
  ) {
    using U = PointerElementType<DestinationPointerType>;

    const short2 position = get_position(thread_context);
    destination += position.y * row_stride + position.x * col_stride;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      const auto row = row_offset + i * THREAD_ELEMENT_ROW_STRIDE;
      const auto col = col_offset;

      if constexpr (metal::is_same_v<ColStride, Int<1>>) {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
          destination[row * row_stride + col + j] =
              static_cast<U>(source[i * THREAD_ELEMENT_COLS + j]);
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
          destination[row * row_stride + (col + j) * col_stride] =
              static_cast<U>(source[i * THREAD_ELEMENT_COLS + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DestinationPointerType,
      typename RowStride,
      typename ColStride,
      typename RowLimit,
      typename ColLimit,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread ThreadVector<T>& source,
      DestinationPointerType destination,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      RowOffset row_offset,
      ColOffset col_offset,
      const thread ThreadContext& thread_context
  ) {
    using U = PointerElementType<DestinationPointerType>;

    const short2 position = get_position(thread_context);
    destination += position.y * row_stride + position.x * col_stride;
    auto local_row_limit = row_limit - position.y;
    auto local_col_limit = col_limit - position.x;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      const auto row = row_offset + i * THREAD_ELEMENT_ROW_STRIDE;
      const auto col = col_offset;

      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
        if (row < local_row_limit && (col + j) < local_col_limit) {
          destination[row * row_stride + (col + j) * col_stride] =
              static_cast<U>(source[i * THREAD_ELEMENT_COLS + j]);
        }
      }
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread ThreadVector<CType>& output_col_0,
      thread ThreadVector<CType>& output_col_1,
      const thread ThreadVector<AType>& left,
      metal::bool_constant<transpose_a>,
      const thread ThreadVector<BType>& right_col_0,
      const thread ThreadVector<BType>& right_col_1,
      metal::bool_constant<transpose_b>
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

    auto cooperative_left =
        matmul_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_right =
        matmul_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_output =
        matmul_op.template get_destination_cooperative_tensor<
            decltype(cooperative_left),
            decltype(cooperative_right),
            CType>();

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      cooperative_left[i] = left[i];
    }

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      cooperative_right[i] = right_col_0[i];
      cooperative_right[ELEMENTS_PER_THREAD + i] = right_col_1[i];
    }

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      cooperative_output[i] = output_col_0[i];
      cooperative_output[ELEMENTS_PER_THREAD + i] = output_col_1[i];
    }

    matmul_op.run(cooperative_left, cooperative_right, cooperative_output);

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      output_col_0[i] = cooperative_output[i];
      output_col_1[i] = cooperative_output[ELEMENTS_PER_THREAD + i];
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread ThreadVector<CType>& output_row_0,
      thread ThreadVector<CType>& output_row_1,
      const thread ThreadVector<AType>& left_row_0,
      const thread ThreadVector<AType>& left_row_1,
      metal::bool_constant<transpose_a>,
      const thread ThreadVector<BType>& right,
      metal::bool_constant<transpose_b>
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

    auto cooperative_left =
        matmul_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_right =
        matmul_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_output =
        matmul_op.template get_destination_cooperative_tensor<
            decltype(cooperative_left),
            decltype(cooperative_right),
            CType>();

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      cooperative_left[i] = left_row_0[i];
      cooperative_left[ELEMENTS_PER_THREAD + i] = left_row_1[i];
    }

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      cooperative_right[i] = right[i];
    }

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      cooperative_output[i] = output_row_0[i];
      cooperative_output[ELEMENTS_PER_THREAD + i] = output_row_1[i];
    }

    matmul_op.run(cooperative_left, cooperative_right, cooperative_output);

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
      output_row_0[i] = cooperative_output[i];
      output_row_1[i] = cooperative_output[ELEMENTS_PER_THREAD + i];
    }
  }

  template <
      bool transpose_a,
      bool transpose_b,
      class OutputTile,
      class LeftTile,
      class RightTile>
  METAL_FUNC static void tile_matmul(
      thread OutputTile& output,
      thread LeftTile& left,
      thread RightTile& right
  ) {
    constexpr ushort left_tile_m =
        transpose_a ? LeftTile::TILE_COLS : LeftTile::TILE_ROWS;
    constexpr ushort tile_m = OutputTile::TILE_ROWS;
    static_assert(
        left_tile_m == tile_m,
        "tile matmul: M dimensions do not match"
    );

    constexpr ushort right_tile_n =
        transpose_b ? RightTile::TILE_ROWS : RightTile::TILE_COLS;
    constexpr ushort tile_n = OutputTile::TILE_COLS;
    static_assert(
        right_tile_n == tile_n,
        "tile matmul: N dimensions do not match"
    );

    constexpr ushort left_tile_k =
        transpose_a ? LeftTile::TILE_ROWS : LeftTile::TILE_COLS;
    constexpr ushort tile_k =
        transpose_b ? RightTile::TILE_COLS : RightTile::TILE_ROWS;
    static_assert(
        left_tile_k == tile_k,
        "tile matmul: K dimensions do not match"
    );

    constexpr auto transpose_left = metal::bool_constant<transpose_a>{};
    constexpr auto transpose_right = metal::bool_constant<transpose_b>{};

    if constexpr (tile_n == 1 && tile_m % 2 == 0) {
      METAL_PRAGMA_UNROLL
      for (ushort row = 0; row < tile_m; row += 2) {
        METAL_PRAGMA_UNROLL
        for (ushort col = 0; col < tile_n; ++col) {
          METAL_PRAGMA_UNROLL
          for (ushort k = 0; k < tile_k; ++k) {
            mma<
                typename OutputTile::ElementType,
                typename LeftTile::ElementType,
                typename RightTile::ElementType,
                transpose_a,
                transpose_b>(
                output.fragment_at(row, col),
                output.fragment_at(row + 1, col),
                left.fragment_at(row, k, transpose_left),
                left.fragment_at(row + 1, k, transpose_left),
                metal::bool_constant<transpose_a>{},
                right.fragment_at(k, col, transpose_right),
                metal::bool_constant<transpose_b>{}
            );
          }
        }
      }
    } else if constexpr (tile_n % 2 == 0) {
      METAL_PRAGMA_UNROLL
      for (ushort row = 0; row < tile_m; ++row) {
        METAL_PRAGMA_UNROLL
        for (ushort col = 0; col < tile_n; col += 2) {
          METAL_PRAGMA_UNROLL
          for (ushort k = 0; k < tile_k; ++k) {
            mma<
                typename OutputTile::ElementType,
                typename LeftTile::ElementType,
                typename RightTile::ElementType,
                transpose_a,
                transpose_b>(
                output.fragment_at(row, col),
                output.fragment_at(row, col + 1),
                left.fragment_at(row, k, transpose_left),
                metal::bool_constant<transpose_a>{},
                right.fragment_at(k, col, transpose_right),
                right.fragment_at(k, col + 1, transpose_right),
                metal::bool_constant<transpose_b>{}
            );
          }
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu

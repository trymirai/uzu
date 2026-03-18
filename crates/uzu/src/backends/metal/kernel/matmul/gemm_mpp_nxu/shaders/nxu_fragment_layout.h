#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "../../common/defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

/// Defines how 32 threads in a simdgroup cooperatively own elements of a 16x16
/// matrix tile using the M5+ native cooperative tensor layout.
///
/// Each thread holds 8 elements arranged as 4 consecutive columns in 2 rows
/// separated by a stride of 8.
///
/// WARNING: This layout ONLY matches the cooperative tensor hardware layout on
/// M5+ Apple Silicon. On pre-M5, cooperative tensors use a different layout and
/// direct linear copy ct[i]=frag[i] produces incorrect results.
struct NxuFragmentLayout {
  METAL_CONST short FRAGMENT_ROWS = 16;
  METAL_CONST short FRAGMENT_COLUMNS = 16;
  METAL_CONST short ELEMENTS_PER_FRAGMENT =
      (FRAGMENT_ROWS * FRAGMENT_COLUMNS) / 32;
  METAL_CONST short ELEMENT_ROWS = 2;
  METAL_CONST short ELEMENT_COLUMNS = 4;
  METAL_CONST short ELEMENT_ROW_STRIDE = 8;

  static_assert(
      ELEMENT_ROWS * ELEMENT_COLUMNS == ELEMENTS_PER_FRAGMENT,
      "Fragment shape is not consistent with fragment element count"
  );

  template <typename U>
  using FragmentVectorType = typename metal::vec<U, ELEMENTS_PER_FRAGMENT>;

  METAL_FUNC static short2 get_coordinate() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short quad_id = simd_lane_id / 4;
    const short fragment_row = ((quad_id / 4) * 4 | ((simd_lane_id / 2) % 4));
    const short fragment_col = ((quad_id % 4 / 2) * 2 | (simd_lane_id % 2)) * 4;
    return short2{fragment_col, fragment_row};
  }

  METAL_FUNC static short2 get_coordinate(short element_index) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short quad_id = simd_lane_id / 4;
    const short fragment_row = ((quad_id / 4) * 4 | ((simd_lane_id / 2) % 4)) +
                               (element_index / 4) * 8;
    const short fragment_col =
        ((quad_id % 4 / 2) * 2 | (simd_lane_id % 2)) * 4 + element_index % 4;
    return short2{fragment_col, fragment_row};
  }

  template <
      typename T,
      typename SourcePointerType,
      typename StrideRow,
      typename StrideCol,
      typename OffsetRow = int,
      typename OffsetCol = int>
  METAL_FUNC static constexpr void load(
      thread FragmentVectorType<T>& destination,
      SourcePointerType source,
      StrideRow stride_row,
      StrideCol stride_col,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0
  ) {
    const short2 simd_coordinate = get_coordinate();
    METAL_PRAGMA_UNROLL
    for (short row_group = 0; row_group < ELEMENT_ROWS; row_group++) {
      const auto row =
          offset_row + row_group * ELEMENT_ROW_STRIDE + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      METAL_PRAGMA_UNROLL
      for (short col_offset = 0; col_offset < ELEMENT_COLUMNS; col_offset++) {
        destination[row_group * ELEMENT_COLUMNS + col_offset] = static_cast<T>(
            source[row * stride_row + (col + col_offset) * stride_col]
        );
      }
    }
  }

  template <
      typename T,
      typename SourcePointerType,
      typename StrideRow,
      typename StrideCol,
      typename RowLimit,
      typename ColumnLimit,
      typename OffsetRow = int,
      typename OffsetCol = int>
  METAL_FUNC static constexpr void load_checked(
      thread FragmentVectorType<T>& destination,
      SourcePointerType source,
      StrideRow stride_row,
      StrideCol stride_col,
      RowLimit row_limit,
      ColumnLimit column_limit,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0
  ) {
    const short2 simd_coordinate = get_coordinate();
    METAL_PRAGMA_UNROLL
    for (short row_group = 0; row_group < ELEMENT_ROWS; row_group++) {
      const auto row =
          offset_row + row_group * ELEMENT_ROW_STRIDE + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      METAL_PRAGMA_UNROLL
      for (short col_offset = 0; col_offset < ELEMENT_COLUMNS; col_offset++) {
        if (row < row_limit && (col + col_offset) < column_limit) {
          destination[row_group * ELEMENT_COLUMNS + col_offset] =
              static_cast<T>(
                  source[row * stride_row + (col + col_offset) * stride_col]
              );
        } else {
          destination[row_group * ELEMENT_COLUMNS + col_offset] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DestinationPointerType,
      typename StrideRow,
      typename StrideCol,
      typename OffsetRow = int,
      typename OffsetCol = int>
  METAL_FUNC static constexpr void store(
      const thread FragmentVectorType<T>& source,
      DestinationPointerType destination,
      StrideRow stride_row,
      StrideCol stride_col,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0
  ) {
    using OutputElementType = PointerElementType<DestinationPointerType>;
    const short2 simd_coordinate = get_coordinate();
    METAL_PRAGMA_UNROLL
    for (short row_group = 0; row_group < ELEMENT_ROWS; row_group++) {
      const auto row =
          offset_row + row_group * ELEMENT_ROW_STRIDE + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      METAL_PRAGMA_UNROLL
      for (short col_offset = 0; col_offset < ELEMENT_COLUMNS; col_offset++) {
        destination[row * stride_row + (col + col_offset) * stride_col] =
            static_cast<OutputElementType>(
                source[row_group * ELEMENT_COLUMNS + col_offset]
            );
      }
    }
  }

  template <
      typename T,
      typename DestinationPointerType,
      typename StrideRow,
      typename StrideCol,
      typename RowLimit,
      typename ColumnLimit,
      typename OffsetRow = int,
      typename OffsetCol = int>
  METAL_FUNC static constexpr void store_checked(
      const thread FragmentVectorType<T>& source,
      DestinationPointerType destination,
      StrideRow stride_row,
      StrideCol stride_col,
      RowLimit row_limit,
      ColumnLimit column_limit,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0
  ) {
    using OutputElementType = PointerElementType<DestinationPointerType>;
    const short2 simd_coordinate = get_coordinate();
    METAL_PRAGMA_UNROLL
    for (short row_group = 0; row_group < ELEMENT_ROWS; row_group++) {
      const auto row =
          offset_row + row_group * ELEMENT_ROW_STRIDE + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      METAL_PRAGMA_UNROLL
      for (short col_offset = 0; col_offset < ELEMENT_COLUMNS; col_offset++) {
        if (row < row_limit && (col + col_offset) < column_limit) {
          destination[row * stride_row + (col + col_offset) * stride_col] =
              static_cast<OutputElementType>(
                  source[row_group * ELEMENT_COLUMNS + col_offset]
              );
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu

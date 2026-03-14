#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

/// Defines how 32 SIMD threads cooperatively own elements of a 16×16 matrix
/// tile.
///
/// Each thread holds 8 elements arranged as 4 consecutive columns in 2 rows
/// separated by a stride of 8. This layout is optimized for coalesced memory
/// access: all 32 threads reading a single row access a contiguous 64-byte
/// span.
///
/// For thread with simd_lane_id=0:
///   elements 0-3 → (row=0, cols=0-3)
///   elements 4-7 → (row=8, cols=0-3)
///
/// WARNING: This layout does NOT match the cooperative tensor layout on pre-M5
/// Apple Silicon. On M2/M3/M4, cooperative tensors use a stride-8 column
/// pattern (cols {0,1,8,9}) while this layout uses consecutive columns (cols
/// {0,1,2,3}). Direct linear copy ct[i]=frag[i] produces incorrect results on
/// pre-M5 chips.
struct SimdgroupFragmentLayout {
  MTL_CONST short FRAGMENT_ROWS = 16;
  MTL_CONST short FRAGMENT_COLUMNS = 16;
  MTL_CONST short ELEMENTS_PER_FRAGMENT =
      (FRAGMENT_ROWS * FRAGMENT_COLUMNS) / 32;
  MTL_CONST short ELEMENT_ROWS = 2;
  MTL_CONST short ELEMENT_COLUMNS = 4;
  MTL_CONST short ELEMENT_ROW_STRIDE = 8;

  static_assert(
      ELEMENT_ROWS * ELEMENT_COLUMNS == ELEMENTS_PER_FRAGMENT,
      "Fragment shape is not consistent with fragment element count"
  );

  template <typename U>
  using FragmentVectorType = typename metal::vec<U, ELEMENTS_PER_FRAGMENT>;

  /// Returns the (column, row) coordinate of this thread's base position in the
  /// 16×16 tile.
  METAL_FUNC static short2 get_coordinate() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short quad_id = simd_lane_id / 4;
    const short fragment_row = ((quad_id / 4) * 4 | ((simd_lane_id / 2) % 4));
    const short fragment_col = ((quad_id % 4 / 2) * 2 | (simd_lane_id % 2)) * 4;
    return short2{fragment_col, fragment_row};
  }

  /// Returns the (column, row) coordinate for a specific element index within
  /// this thread's fragment.
  METAL_FUNC static short2 get_coordinate(short element_index) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short quad_id = simd_lane_id / 4;
    const short fragment_row = ((quad_id / 4) * 4 | ((simd_lane_id / 2) % 4)) +
                               (element_index / 4) * 8;
    const short fragment_col =
        ((quad_id % 4 / 2) * 2 | (simd_lane_id % 2)) * 4 + element_index % 4;
    return short2{fragment_col, fragment_row};
  }

  /// Loads a 16×16 tile from memory into this thread's fragment register.
  template <
      typename T,
      typename SourcePointerType,
      typename StrideRow,
      typename StrideCol,
      typename OffsetRow = int,
      typename OffsetCol = int
  >
  METAL_FUNC static constexpr void load(
      thread FragmentVectorType<T>& destination,
      SourcePointerType source,
      StrideRow stride_row,
      StrideCol stride_col,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0
  ) {
    const short2 simd_coordinate = get_coordinate();
    PRAGMA_UNROLL
    for (short row_group = 0; row_group < ELEMENT_ROWS; row_group++) {
      const auto row =
          offset_row + row_group * ELEMENT_ROW_STRIDE + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      PRAGMA_UNROLL
      for (short col_offset = 0; col_offset < ELEMENT_COLUMNS; col_offset++) {
        destination[row_group * ELEMENT_COLUMNS + col_offset] = static_cast<T>(
            source[row * stride_row + (col + col_offset) * stride_col]
        );
      }
    }
  }

  /// Loads a 16×16 tile with bounds checking on both row and column dimensions.
  template <
      typename T,
      typename SourcePointerType,
      typename StrideRow,
      typename StrideCol,
      typename RowLimit,
      typename ColumnLimit,
      typename OffsetRow = int,
      typename OffsetCol = int
  >
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
    PRAGMA_UNROLL
    for (short row_group = 0; row_group < ELEMENT_ROWS; row_group++) {
      const auto row =
          offset_row + row_group * ELEMENT_ROW_STRIDE + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      PRAGMA_UNROLL
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

  /// Stores a fragment register to a 16×16 tile in memory.
  template <
      typename T,
      typename DestinationPointerType,
      typename StrideRow,
      typename StrideCol,
      typename OffsetRow = int,
      typename OffsetCol = int
  >
  METAL_FUNC static constexpr void store(
      const thread FragmentVectorType<T>& source,
      DestinationPointerType destination,
      StrideRow stride_row,
      StrideCol stride_col,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0
  ) {
    using OutputElementType = pointer_element_t<DestinationPointerType>;
    const short2 simd_coordinate = get_coordinate();
    PRAGMA_UNROLL
    for (short row_group = 0; row_group < ELEMENT_ROWS; row_group++) {
      const auto row =
          offset_row + row_group * ELEMENT_ROW_STRIDE + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      PRAGMA_UNROLL
      for (short col_offset = 0; col_offset < ELEMENT_COLUMNS; col_offset++) {
        destination[row * stride_row + (col + col_offset) * stride_col] =
            static_cast<OutputElementType>(
                source[row_group * ELEMENT_COLUMNS + col_offset]
            );
      }
    }
  }

  /// Stores a fragment register with bounds checking on both row and column
  /// dimensions.
  template <
      typename T,
      typename DestinationPointerType,
      typename StrideRow,
      typename StrideCol,
      typename RowLimit,
      typename ColumnLimit,
      typename OffsetRow = int,
      typename OffsetCol = int
  >
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
    using OutputElementType = pointer_element_t<DestinationPointerType>;
    const short2 simd_coordinate = get_coordinate();
    PRAGMA_UNROLL
    for (short row_group = 0; row_group < ELEMENT_ROWS; row_group++) {
      const auto row =
          offset_row + row_group * ELEMENT_ROW_STRIDE + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      PRAGMA_UNROLL
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

#pragma once

#include "nxu_fragment_layout.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

namespace uzu {
namespace matmul {

// M5+-only types and functions that use NxuFragmentLayout to load data into
// registers, then copy linearly to cooperative tensors (ct[i] = frag[i]).
// On pre-M5, use cooperative_tensor_gemm instead.

template <
    typename T,
    short ROWS_,
    short COLS_,
    typename FragmentLayout = NxuFragmentLayout>
struct NxuSubTile {
  METAL_CONST short ROWS = ROWS_;
  METAL_CONST short COLS = COLS_;
  METAL_CONST short FRAGMENT_ROWS = FragmentLayout::FRAGMENT_ROWS;
  METAL_CONST short FRAGMENT_COLUMNS = FragmentLayout::FRAGMENT_COLUMNS;
  METAL_CONST short ELEMENTS_PER_FRAGMENT =
      FragmentLayout::ELEMENTS_PER_FRAGMENT;
  METAL_CONST short SUBTILE_ROWS = ROWS / FRAGMENT_ROWS;
  METAL_CONST short SUBTILE_COLS = COLS / FRAGMENT_COLUMNS;
  METAL_CONST short NUM_FRAGMENTS = SUBTILE_ROWS * SUBTILE_COLS;
  METAL_CONST short ELEMENTS_PER_SUBTILE =
      NUM_FRAGMENTS * ELEMENTS_PER_FRAGMENT;
  METAL_CONST int ROWS_PER_THREAD = SUBTILE_ROWS * FragmentLayout::ELEMENT_ROWS;
  METAL_CONST int COLS_PER_THREAD =
      SUBTILE_COLS * FragmentLayout::ELEMENT_COLUMNS;

  using ElementType = T;
  using FragmentType = typename FragmentLayout::template FragmentVectorType<T>;
  FragmentType value_fragments[NUM_FRAGMENTS];

  METAL_FUNC constexpr void clear() {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < NUM_FRAGMENTS; ++i) {
      value_fragments[i] = FragmentType(0);
    }
  }

  METAL_FUNC constexpr thread FragmentType& fragment_at(
      const short row,
      const short col
  ) {
    return value_fragments[row * SUBTILE_COLS + col];
  }

  METAL_FUNC constexpr const thread FragmentType& fragment_at(
      const short row,
      const short col
  ) const {
    return value_fragments[row * SUBTILE_COLS + col];
  }

  template <bool transpose>
  METAL_FUNC constexpr thread FragmentType& fragment_at(
      const short row,
      const short col,
      metal::bool_constant<transpose>
  ) {
    return transpose ? fragment_at(col, row) : fragment_at(row, col);
  }

  template <bool transpose>
  METAL_FUNC constexpr const thread FragmentType& fragment_at(
      const short row,
      const short col,
      metal::bool_constant<transpose>
  ) const {
    return transpose ? fragment_at(col, row) : fragment_at(row, col);
  }

  METAL_FUNC thread T* elements() {
    return reinterpret_cast<thread T*>(value_fragments);
  }
  METAL_FUNC const thread T* elements() const {
    return reinterpret_cast<const thread T*>(value_fragments);
  }

  template <
      typename SourcePointerType,
      typename StrideRow,
      typename StrideCol,
      typename OffsetRow = int,
      typename OffsetCol = int>
  METAL_FUNC constexpr void load(
      SourcePointerType source,
      StrideRow stride_row,
      StrideCol stride_col,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0
  ) {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < SUBTILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < SUBTILE_COLS; ++j) {
        FragmentLayout::load(
            fragment_at(i, j),
            source,
            stride_row,
            stride_col,
            offset_row + i * FRAGMENT_ROWS,
            offset_col + j * FRAGMENT_COLUMNS
        );
      }
    }
  }

  template <
      typename DestinationPointerType,
      typename StrideRow,
      typename StrideCol,
      typename OffsetRow = int,
      typename OffsetCol = int>
  METAL_FUNC constexpr void store(
      DestinationPointerType destination,
      StrideRow stride_row,
      StrideCol stride_col,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0
  ) const {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < SUBTILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < SUBTILE_COLS; ++j) {
        FragmentLayout::store(
            fragment_at(i, j),
            destination,
            stride_row,
            stride_col,
            offset_row + i * FRAGMENT_ROWS,
            offset_col + j * FRAGMENT_COLUMNS
        );
      }
    }
  }

  template <
      typename SourcePointerType,
      typename StrideRow,
      typename StrideCol,
      typename RowLimit,
      typename ColumnLimit,
      typename OffsetRow = int,
      typename OffsetCol = int>
  METAL_FUNC constexpr void load_checked(
      SourcePointerType source,
      StrideRow stride_row,
      StrideCol stride_col,
      RowLimit row_limit,
      ColumnLimit column_limit,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0
  ) {
    METAL_PRAGMA_UNROLL
    for (int i = 0; i < SUBTILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (int j = 0; j < SUBTILE_COLS; ++j) {
        FragmentLayout::load_checked(
            fragment_at(i, j),
            source,
            stride_row,
            stride_col,
            row_limit,
            column_limit,
            offset_row + (i * FRAGMENT_ROWS),
            offset_col + (j * FRAGMENT_COLUMNS)
        );
      }
    }
  }

  template <
      typename DestinationPointerType,
      typename StrideRow,
      typename StrideCol,
      typename RowLimit,
      typename ColumnLimit,
      typename OffsetRow = int,
      typename OffsetCol = int>
  METAL_FUNC constexpr void store_checked(
      DestinationPointerType destination,
      StrideRow stride_row,
      StrideCol stride_col,
      RowLimit row_limit,
      ColumnLimit column_limit,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0
  ) const {
    METAL_PRAGMA_UNROLL
    for (int i = 0; i < SUBTILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (int j = 0; j < SUBTILE_COLS; ++j) {
        FragmentLayout::store_checked(
            fragment_at(i, j),
            destination,
            stride_row,
            stride_col,
            row_limit,
            column_limit,
            offset_row + (i * FRAGMENT_ROWS),
            offset_col + (j * FRAGMENT_COLUMNS)
        );
      }
    }
  }
};

// 16x32x16 paired-N matmul: one A fragment, two B fragments, two C fragments.
// Satisfies MPP constraint that at least one of M,N,K must be 32.
template <
    typename AccumulatorType,
    typename LeftType,
    typename RightType,
    bool transpose_left,
    bool transpose_right,
    typename FragmentLayout = NxuFragmentLayout>
METAL_FUNC void nxu_paired_matmul(
    thread
    typename FragmentLayout::template FragmentVectorType<AccumulatorType>&
        c_frag_0,
    thread
    typename FragmentLayout::template FragmentVectorType<AccumulatorType>&
        c_frag_1,
    const thread
    typename FragmentLayout::template FragmentVectorType<LeftType>& a_frag,
    metal::bool_constant<transpose_left>,
    const thread
    typename FragmentLayout::template FragmentVectorType<RightType>& b_frag_0,
    const thread
    typename FragmentLayout::template FragmentVectorType<RightType>& b_frag_1,
    metal::bool_constant<transpose_right>
) {

  constexpr short ELEMENTS_PER_FRAGMENT = FragmentLayout::ELEMENTS_PER_FRAGMENT;

  constexpr auto matmul_descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      transpose_left,
      transpose_right,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
  );

  mpp::tensor_ops::matmul2d<matmul_descriptor, metal::execution_simdgroup>
      matmul_operation;

  auto left_tensor =
      matmul_operation.template get_left_input_cooperative_tensor<
          LeftType,
          RightType,
          AccumulatorType>();
  auto right_tensor =
      matmul_operation.template get_right_input_cooperative_tensor<
          LeftType,
          RightType,
          AccumulatorType>();
  auto accumulator_tensor =
      matmul_operation.template get_destination_cooperative_tensor<
          decltype(left_tensor),
          decltype(right_tensor),
          AccumulatorType>();

  METAL_PRAGMA_UNROLL
  for (short i = 0; i < ELEMENTS_PER_FRAGMENT; i++) {
    left_tensor[i] = a_frag[i];
  }

  METAL_PRAGMA_UNROLL
  for (short i = 0; i < ELEMENTS_PER_FRAGMENT; i++) {
    right_tensor[i] = b_frag_0[i];
    right_tensor[ELEMENTS_PER_FRAGMENT + i] = b_frag_1[i];
  }

  METAL_PRAGMA_UNROLL
  for (short i = 0; i < ELEMENTS_PER_FRAGMENT; i++) {
    accumulator_tensor[i] = c_frag_0[i];
    accumulator_tensor[ELEMENTS_PER_FRAGMENT + i] = c_frag_1[i];
  }

  matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);

  METAL_PRAGMA_UNROLL
  for (short i = 0; i < ELEMENTS_PER_FRAGMENT; i++) {
    c_frag_0[i] = accumulator_tensor[i];
    c_frag_1[i] = accumulator_tensor[ELEMENTS_PER_FRAGMENT + i];
  }
}

template <
    typename T,
    short TILE_ROWS_,
    short TILE_COLS_,
    class SubtileType = NxuSubTile<T, 16, 16>>
struct NxuTile {
  using SubTileType = SubtileType;
  using ElementType = T;

  METAL_CONST short SUBTILE_ROWS = SubTileType::ROWS;
  METAL_CONST short SUBTILE_COLS = SubTileType::COLS;
  METAL_CONST short ELEMENTS_PER_SUBTILE = SubTileType::ELEMENTS_PER_SUBTILE;
  METAL_CONST short TILE_ROWS = TILE_ROWS_;
  METAL_CONST short TILE_COLS = TILE_COLS_;
  METAL_CONST short ROWS = TILE_ROWS * SUBTILE_ROWS;
  METAL_CONST short COLS = TILE_COLS * SUBTILE_COLS;
  METAL_CONST short SUBTILES = TILE_ROWS * TILE_COLS;
  METAL_CONST short ELEMENTS_PER_TILE = SUBTILES * ELEMENTS_PER_SUBTILE;

  SubtileType value_subtiles[SUBTILES];

  METAL_FUNC NxuTile() thread {}

  METAL_FUNC constexpr void clear() {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < SUBTILES; ++i) {
      value_subtiles[i].clear();
    }
  }

  METAL_FUNC constexpr thread SubtileType& subtile_at(
      const short row,
      const short col
  ) {
    return value_subtiles[row * TILE_COLS + col];
  }

  METAL_FUNC constexpr const thread SubtileType& subtile_at(
      const short row,
      const short col
  ) const {
    return value_subtiles[row * TILE_COLS + col];
  }

  template <typename U>
  METAL_FUNC void load(const device U* source, const int leading_dimension) {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; ++j) {
        subtile_at(i, j).load(
            &source[(i * SUBTILE_ROWS * leading_dimension + j * SUBTILE_COLS)],
            leading_dimension,
            1
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void store(
      device U* destination,
      const int leading_dimension
  ) const {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; ++j) {
        subtile_at(i, j).store(
            &destination
                [(i * SUBTILE_ROWS * leading_dimension + j * SUBTILE_COLS)],
            leading_dimension,
            1
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void load_checked(
      const device U* source,
      const int leading_dimension,
      const short2 tile_dimensions
  ) {
    METAL_PRAGMA_UNROLL
    for (int i = 0; i < TILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (int j = 0; j < TILE_COLS; ++j) {
        subtile_at(i, j).load_checked(
            source,
            leading_dimension,
            1,
            tile_dimensions.y,
            tile_dimensions.x,
            i * SUBTILE_ROWS,
            j * SUBTILE_COLS
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void store_checked(
      device U* destination,
      const int leading_dimension,
      const short2 tile_dimensions
  ) const {
    METAL_PRAGMA_UNROLL
    for (int i = 0; i < TILE_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (int j = 0; j < TILE_COLS; ++j) {
        subtile_at(i, j).store_checked(
            destination,
            leading_dimension,
            1,
            tile_dimensions.y,
            tile_dimensions.x,
            i * SUBTILE_ROWS,
            j * SUBTILE_COLS
        );
      }
    }
  }
};

// Tile-level matmul that pairs fragments along N to satisfy the 32-minimum
// constraint. Requires TILE_COLS to be even.
template <
    class AccumulatorTile,
    class LeftTile,
    class RightTile,
    bool transpose_left,
    bool transpose_right>
METAL_FUNC void nxu_tiled_matmul(
    thread AccumulatorTile& accumulator,
    thread LeftTile& left_input,
    metal::bool_constant<transpose_left>,
    thread RightTile& right_input,
    metal::bool_constant<transpose_right>
) {

  constexpr short tiles_m = AccumulatorTile::TILE_ROWS;
  constexpr short tiles_n = AccumulatorTile::TILE_COLS;
  constexpr short tiles_k =
      transpose_left ? LeftTile::TILE_ROWS : LeftTile::TILE_COLS;

  static_assert(
      tiles_n % 2 == 0,
      "NxuTile N dimension must be even for paired 16x32x16 matmul"
  );

  constexpr auto transpose_a_tag = metal::bool_constant<transpose_left>{};
  constexpr auto transpose_b_tag = metal::bool_constant<transpose_right>{};

  METAL_PRAGMA_UNROLL
  for (short i = 0; i < tiles_m; ++i) {
    METAL_PRAGMA_UNROLL
    for (short j = 0; j < tiles_n; j += 2) {
      METAL_PRAGMA_UNROLL
      for (short k = 0; k < tiles_k; ++k) {
        const short a_row = transpose_left ? k : i;
        const short a_col = transpose_left ? i : k;
        const short b_row_0 = transpose_right ? j : k;
        const short b_col_0 = transpose_right ? k : j;
        const short b_row_1 = transpose_right ? (j + 1) : k;
        const short b_col_1 = transpose_right ? k : (j + 1);
        nxu_paired_matmul<
            typename AccumulatorTile::SubTileType::ElementType,
            typename LeftTile::SubTileType::ElementType,
            typename RightTile::SubTileType::ElementType,
            transpose_left,
            transpose_right>(
            accumulator.subtile_at(i, j).fragment_at(0, 0),
            accumulator.subtile_at(i, j + 1).fragment_at(0, 0),
            left_input.subtile_at(a_row, a_col).fragment_at(0, 0),
            transpose_a_tag,
            right_input.subtile_at(b_row_0, b_col_0).fragment_at(0, 0),
            right_input.subtile_at(b_row_1, b_col_1).fragment_at(0, 0),
            transpose_b_tag
        );
      }
    }
  }
}

} // namespace matmul
} // namespace uzu

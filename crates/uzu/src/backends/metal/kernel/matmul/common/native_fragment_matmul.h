#pragma once

#include "simdgroup_fragment_layout.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

namespace uzu {
namespace matmul {

// These types and functions use SimdgroupFragmentLayout to load data into
// registers, then copy linearly to cooperative tensors (ct[i] = frag[i]).
// This is ONLY correct on M5+ where the hardware cooperative tensor layout
// matches SimdgroupFragmentLayout. On pre-M5, use cooperative_tensor_gemm instead.

/// A grid of 16×16 SimdgroupFragmentLayout fragments forming a subtile.
template <
    typename T,
    short ROWS_,
    short COLS_,
    typename FragmentLayout = SimdgroupFragmentLayout>
struct MppSubTile {
  MTL_CONST short ROWS = ROWS_;
  MTL_CONST short COLS = COLS_;
  MTL_CONST short FRAGMENT_ROWS = FragmentLayout::FRAGMENT_ROWS;
  MTL_CONST short FRAGMENT_COLUMNS = FragmentLayout::FRAGMENT_COLUMNS;
  MTL_CONST short ELEMENTS_PER_FRAGMENT = FragmentLayout::ELEMENTS_PER_FRAGMENT;
  MTL_CONST short SUBTILE_ROWS = ROWS / FRAGMENT_ROWS;
  MTL_CONST short SUBTILE_COLS = COLS / FRAGMENT_COLUMNS;
  MTL_CONST short NUM_FRAGMENTS = SUBTILE_ROWS * SUBTILE_COLS;
  MTL_CONST short ELEMENTS_PER_SUBTILE = NUM_FRAGMENTS * ELEMENTS_PER_FRAGMENT;
  MTL_CONST int ROWS_PER_THREAD = SUBTILE_ROWS * FragmentLayout::ELEMENT_ROWS;
  MTL_CONST int COLS_PER_THREAD = SUBTILE_COLS * FragmentLayout::ELEMENT_COLUMNS;
  MTL_CONST short FRAGMENT_THREAD_ROWS = FragmentLayout::ELEMENT_ROWS;
  MTL_CONST short FRAGMENT_THREAD_COLUMNS = FragmentLayout::ELEMENT_COLUMNS;
  MTL_CONST short FRAGMENT_ROW_STRIDE = FragmentLayout::ELEMENT_ROW_STRIDE;

  using FragmentType = typename FragmentLayout::template FragmentVectorType<T>;
  FragmentType value_fragments[NUM_FRAGMENTS];

  METAL_FUNC constexpr void clear() {
    PRAGMA_UNROLL
    for (short i = 0; i < NUM_FRAGMENTS; ++i) {
      value_fragments[i] = FragmentType(0);
    }
  }

  METAL_FUNC constexpr thread FragmentType& fragment_at(const short row, const short col) {
    return value_fragments[row * SUBTILE_COLS + col];
  }

  METAL_FUNC constexpr const thread FragmentType& fragment_at(const short row, const short col) const {
    return value_fragments[row * SUBTILE_COLS + col];
  }

  METAL_FUNC thread T* elements() { return reinterpret_cast<thread T*>(value_fragments); }
  METAL_FUNC const thread T* elements() const { return reinterpret_cast<const thread T*>(value_fragments); }

  template <typename SourcePointerType, typename StrideRow, typename StrideCol, typename OffsetRow = int, typename OffsetCol = int>
  METAL_FUNC constexpr void load(SourcePointerType source, StrideRow stride_row, StrideCol stride_col, OffsetRow offset_row = 0, OffsetCol offset_col = 0) {
    PRAGMA_UNROLL
    for (short i = 0; i < SUBTILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (short j = 0; j < SUBTILE_COLS; ++j) {
        FragmentLayout::load(fragment_at(i, j), source, stride_row, stride_col, offset_row + i * FRAGMENT_ROWS, offset_col + j * FRAGMENT_COLUMNS);
      }
    }
  }

  template <typename DestinationPointerType, typename StrideRow, typename StrideCol, typename OffsetRow = int, typename OffsetCol = int>
  METAL_FUNC constexpr void store(DestinationPointerType destination, StrideRow stride_row, StrideCol stride_col, OffsetRow offset_row = 0, OffsetCol offset_col = 0) const {
    PRAGMA_UNROLL
    for (short i = 0; i < SUBTILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (short j = 0; j < SUBTILE_COLS; ++j) {
        FragmentLayout::store(fragment_at(i, j), destination, stride_row, stride_col, offset_row + i * FRAGMENT_ROWS, offset_col + j * FRAGMENT_COLUMNS);
      }
    }
  }

  template <typename SourcePointerType, typename StrideRow, typename StrideCol, typename RowLimit, typename ColumnLimit, typename OffsetRow = int, typename OffsetCol = int>
  METAL_FUNC constexpr void load_checked(SourcePointerType source, StrideRow stride_row, StrideCol stride_col, RowLimit row_limit, ColumnLimit column_limit, OffsetRow offset_row = 0, OffsetCol offset_col = 0) {
    PRAGMA_UNROLL
    for (int i = 0; i < SUBTILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (int j = 0; j < SUBTILE_COLS; ++j) {
        FragmentLayout::load_checked(fragment_at(i, j), source, stride_row, stride_col, row_limit, column_limit, offset_row + (i * FRAGMENT_ROWS), offset_col + (j * FRAGMENT_COLUMNS));
      }
    }
  }

  template <typename DestinationPointerType, typename StrideRow, typename StrideCol, typename RowLimit, typename ColumnLimit, typename OffsetRow = int, typename OffsetCol = int>
  METAL_FUNC constexpr void store_checked(DestinationPointerType destination, StrideRow stride_row, StrideCol stride_col, RowLimit row_limit, ColumnLimit column_limit, OffsetRow offset_row = 0, OffsetCol offset_col = 0) const {
    PRAGMA_UNROLL
    for (int i = 0; i < SUBTILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (int j = 0; j < SUBTILE_COLS; ++j) {
        FragmentLayout::store_checked(fragment_at(i, j), destination, stride_row, stride_col, row_limit, column_limit, offset_row + (i * FRAGMENT_ROWS), offset_col + (j * FRAGMENT_COLUMNS));
      }
    }
  }
};

/// Performs a single subtile matrix multiply-accumulate using cooperative tensors.
/// Copies data linearly from SimdgroupFragmentLayout registers to cooperative tensor indices.
/// ONLY correct on M5+ where the cooperative tensor layout matches SimdgroupFragmentLayout.
template <
    short accumulator_rows, short accumulator_cols,
    short left_rows, short left_cols,
    short right_rows, short right_cols,
    typename AccumulatorType, typename LeftType, typename RightType,
    bool transpose_left, bool transpose_right,
    typename FragmentLayout = SimdgroupFragmentLayout>
METAL_FUNC void native_fragment_matmul(
    thread MppSubTile<AccumulatorType, accumulator_rows, accumulator_cols, FragmentLayout>& accumulator,
    thread MppSubTile<LeftType, left_rows, left_cols, FragmentLayout>& left_input,
    metal::bool_constant<transpose_left>,
    thread MppSubTile<RightType, right_rows, right_cols, FragmentLayout>& right_input,
    metal::bool_constant<transpose_right>) {

  constexpr short m_dimension_left = transpose_left ? left_cols : left_rows;
  constexpr short m_dimension_accumulator = accumulator_rows;
  static_assert(m_dimension_left == m_dimension_accumulator, "MPP matmul: M dimensions do not match");

  constexpr short n_dimension_right = transpose_right ? right_rows : right_cols;
  constexpr short n_dimension_accumulator = accumulator_cols;
  static_assert(n_dimension_right == n_dimension_accumulator, "MPP matmul: N dimensions do not match");

  constexpr short k_dimension_left = transpose_left ? left_rows : left_cols;
  constexpr short k_dimension_right = transpose_right ? right_cols : right_rows;
  static_assert(k_dimension_left == k_dimension_right, "MPP matmul: K dimensions do not match");

  constexpr short total_m = m_dimension_accumulator;
  constexpr short total_n = n_dimension_accumulator;
  constexpr short total_k = k_dimension_left;

  constexpr int tiles_m = total_m / 16;
  constexpr int tiles_n = total_n / 16;
  constexpr int tiles_k = total_k / 16;

  constexpr auto matmul_descriptor = mpp::tensor_ops::matmul2d_descriptor(
      total_m, total_n, total_k,
      transpose_left, transpose_right, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  mpp::tensor_ops::matmul2d<matmul_descriptor, metal::execution_simdgroup> matmul_operation;

  auto left_tensor = matmul_operation.template get_left_input_cooperative_tensor<LeftType, RightType, AccumulatorType>();
  auto right_tensor = matmul_operation.template get_right_input_cooperative_tensor<LeftType, RightType, AccumulatorType>();
  auto accumulator_tensor = matmul_operation.template get_destination_cooperative_tensor<
      decltype(left_tensor), decltype(right_tensor), AccumulatorType>();

  PRAGMA_UNROLL
  for (short tile_m = 0; tile_m < tiles_m; tile_m++) {
    PRAGMA_UNROLL
    for (short tile_k = 0; tile_k < tiles_k; tile_k++) {
      const short fragment_row = transpose_left ? tile_k : tile_m;
      const short fragment_col = transpose_left ? tile_m : tile_k;
      PRAGMA_UNROLL
      for (short element = 0; element < 8; element++) {
        left_tensor[(tiles_k * tile_m + tile_k) * 8 + element] = left_input.fragment_at(fragment_row, fragment_col)[element];
      }
    }
  }

  PRAGMA_UNROLL
  for (short tile_n = 0; tile_n < tiles_n; tile_n++) {
    PRAGMA_UNROLL
    for (short tile_k = 0; tile_k < tiles_k; tile_k++) {
      const short fragment_row = transpose_right ? tile_n : tile_k;
      const short fragment_col = transpose_right ? tile_k : tile_n;
      PRAGMA_UNROLL
      for (short element = 0; element < 8; element++) {
        right_tensor[(tiles_n * tile_k + tile_n) * 8 + element] = right_input.fragment_at(fragment_row, fragment_col)[element];
      }
    }
  }

  PRAGMA_UNROLL
  for (short i = 0; i < accumulator_tensor.get_capacity(); i++) {
    accumulator_tensor[i] = accumulator.elements()[i];
  }

  matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);

  PRAGMA_UNROLL
  for (short i = 0; i < accumulator_tensor.get_capacity(); i++) {
    accumulator.elements()[i] = accumulator_tensor[i];
  }
}

/// A grid of MppSubTile elements forming a larger tile.
template <typename T, short TILE_ROWS_, short TILE_COLS_, class SubtileType>
struct MppTile {
  using SubTileType = SubtileType;
  using ElementType = T;

  MTL_CONST short SUBTILE_ROWS = SubTileType::ROWS;
  MTL_CONST short SUBTILE_COLS = SubTileType::COLS;
  MTL_CONST short ELEMENTS_PER_SUBTILE = SubTileType::ELEMENTS_PER_SUBTILE;
  MTL_CONST short TILE_ROWS = TILE_ROWS_;
  MTL_CONST short TILE_COLS = TILE_COLS_;
  MTL_CONST short ROWS = TILE_ROWS * SUBTILE_ROWS;
  MTL_CONST short COLS = TILE_COLS * SUBTILE_COLS;
  MTL_CONST short SUBTILES = TILE_ROWS * TILE_COLS;
  MTL_CONST short ELEMENTS_PER_TILE = SUBTILES * ELEMENTS_PER_SUBTILE;
  MTL_CONST short ROWS_PER_THREAD = TILE_ROWS * SubTileType::ROWS_PER_THREAD;
  MTL_CONST short COLS_PER_THREAD = TILE_COLS * SubTileType::COLS_PER_THREAD;

  SubtileType value_subtiles[SUBTILES];

  METAL_FUNC MppTile() thread {}

  METAL_FUNC constexpr void clear() {
    PRAGMA_UNROLL
    for (short i = 0; i < SUBTILES; ++i) { value_subtiles[i].clear(); }
  }

  METAL_FUNC constexpr thread SubtileType& subtile_at(const short row, const short col) {
    return value_subtiles[row * TILE_COLS + col];
  }

  METAL_FUNC constexpr const thread SubtileType& subtile_at(const short row, const short col) const {
    return value_subtiles[row * TILE_COLS + col];
  }

  METAL_FUNC thread ElementType* elements() { return reinterpret_cast<thread ElementType*>(value_subtiles[0].elements()); }
  METAL_FUNC const thread ElementType* elements() const { return reinterpret_cast<const thread ElementType*>(value_subtiles[0].elements()); }

  template <typename U>
  METAL_FUNC void load(const device U* source, const int leading_dimension) {
    PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; ++j) {
        subtile_at(i, j).load(&source[(i * SUBTILE_ROWS * leading_dimension + j * SUBTILE_COLS)], leading_dimension, 1);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store(device U* destination, const int leading_dimension) const {
    PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; ++j) {
        subtile_at(i, j).store(&destination[(i * SUBTILE_ROWS * leading_dimension + j * SUBTILE_COLS)], leading_dimension, 1);
      }
    }
  }

  template <typename U>
  METAL_FUNC void load_checked(const device U* source, const int leading_dimension, const short2 tile_dimensions) {
    PRAGMA_UNROLL
    for (int i = 0; i < TILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (int j = 0; j < TILE_COLS; ++j) {
        subtile_at(i, j).load_checked(source, leading_dimension, 1, tile_dimensions.y, tile_dimensions.x, i * SUBTILE_ROWS, j * SUBTILE_COLS);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store_checked(device U* destination, const int leading_dimension, const short2 tile_dimensions) const {
    PRAGMA_UNROLL
    for (int i = 0; i < TILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (int j = 0; j < TILE_COLS; ++j) {
        subtile_at(i, j).store_checked(destination, leading_dimension, 1, tile_dimensions.y, tile_dimensions.x, i * SUBTILE_ROWS, j * SUBTILE_COLS);
      }
    }
  }
};

/// Performs tiled matrix multiply-accumulate across all subtiles within a tile.
/// Iterates over M, N, K tile indices and calls native_fragment_matmul for each subtile combination.
/// ONLY correct on M5+ (see native_fragment_matmul documentation).
template <class AccumulatorTile, class LeftTile, class RightTile, bool transpose_left, bool transpose_right>
METAL_FUNC void native_tiled_matmul(
    thread AccumulatorTile& accumulator,
    thread LeftTile& left_input,
    metal::bool_constant<transpose_left>,
    thread RightTile& right_input,
    metal::bool_constant<transpose_right>) {

  constexpr short tiles_m = AccumulatorTile::TILE_ROWS;
  constexpr short tiles_n = AccumulatorTile::TILE_COLS;
  constexpr short tiles_k = transpose_left ? LeftTile::TILE_ROWS : LeftTile::TILE_COLS;

  PRAGMA_UNROLL
  for (short i = 0; i < tiles_m; ++i) {
    PRAGMA_UNROLL
    for (short j = 0; j < tiles_n; ++j) {
      PRAGMA_UNROLL
      for (short k = 0; k < tiles_k; ++k) {
        const short left_row = transpose_left ? k : i;
        const short left_col = transpose_left ? i : k;
        const short right_row = transpose_right ? j : k;
        const short right_col = transpose_right ? k : j;
        native_fragment_matmul(
            accumulator.subtile_at(i, j),
            left_input.subtile_at(left_row, left_col),
            metal::bool_constant<transpose_left>{},
            right_input.subtile_at(right_row, right_col),
            metal::bool_constant<transpose_right>{});
      }
    }
  }
}

} // namespace matmul
} // namespace uzu

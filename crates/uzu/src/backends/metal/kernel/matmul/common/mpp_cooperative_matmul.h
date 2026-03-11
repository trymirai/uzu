#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "defines.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace uzu {
namespace matmul {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Section 1: SimdgroupFragmentLayout
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Section 2: Pre-M5 Cooperative Tensor GEMM
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Section 3: M5 Native Fragment Path
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

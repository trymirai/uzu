#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "steel/defines.h"
#include "steel/utils/type_traits.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace uzu {
namespace matmul {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Section 1: SimdgroupFragmentLayout
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Defines how 32 SIMD threads cooperatively own elements of a 16×16 matrix tile.
///
/// Each thread holds 8 elements arranged as 4 consecutive columns in 2 rows
/// separated by a stride of 8. This layout is optimized for coalesced memory
/// access: all 32 threads reading a single row access a contiguous 64-byte span.
///
/// For thread with simd_lane_id=0:
///   elements 0-3 → (row=0, cols=0-3)
///   elements 4-7 → (row=8, cols=0-3)
///
/// WARNING: This layout does NOT match the cooperative tensor layout on pre-M5
/// Apple Silicon. On M2/M3/M4, cooperative tensors use a stride-8 column pattern
/// (cols {0,1,8,9}) while this layout uses consecutive columns (cols {0,1,2,3}).
/// Direct linear copy ct[i]=frag[i] produces incorrect results on pre-M5 chips.
struct SimdgroupFragmentLayout {
  STEEL_CONST short kFragmentRows = 16;
  STEEL_CONST short kFragmentColumns = 16;
  STEEL_CONST short kElementsPerFragment = (kFragmentRows * kFragmentColumns) / 32;
  STEEL_CONST short kElementRows = 2;
  STEEL_CONST short kElementColumns = 4;
  STEEL_CONST short kElementRowStride = 8;

  static_assert(
      kElementRows * kElementColumns == kElementsPerFragment,
      "Fragment shape is not consistent with fragment element count");

  template <typename U>
  using fragment_vector_t = typename metal::vec<U, kElementsPerFragment>;

  /// Returns the (column, row) coordinate of this thread's base position in the 16×16 tile.
  METAL_FUNC static short2 get_coordinate() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short quad_id = simd_lane_id >> 2;
    const short fragment_row = ((quad_id & 4) | ((simd_lane_id >> 1) & 3));
    const short fragment_col = ((quad_id & 2) | (simd_lane_id & 1)) * 4;
    return short2{fragment_col, fragment_row};
  }

  /// Returns the (column, row) coordinate for a specific element index within this thread's fragment.
  METAL_FUNC static short2 get_coordinate(short element_index) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short quad_id = simd_lane_id >> 2;
    const short fragment_row = ((quad_id & 4) | ((simd_lane_id >> 1) & 3)) + (element_index >> 2) * 8;
    const short fragment_col = ((quad_id & 2) | (simd_lane_id & 1)) * 4 + element_index % 4;
    return short2{fragment_col, fragment_row};
  }

  /// Loads a 16×16 tile from memory into this thread's fragment register.
  template <
      typename T,
      typename SourcePointerType,
      typename StrideRow,
      typename StrideCol,
      typename OffsetRow = int,
      typename OffsetCol = int>
  METAL_FUNC static constexpr void load(
      thread fragment_vector_t<T>& destination,
      SourcePointerType source,
      StrideRow stride_row,
      StrideCol stride_col,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0) {
    const short2 simd_coordinate = get_coordinate();
    STEEL_PRAGMA_UNROLL
    for (short row_group = 0; row_group < kElementRows; row_group++) {
      const auto row = offset_row + row_group * kElementRowStride + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      STEEL_PRAGMA_UNROLL
      for (short col_offset = 0; col_offset < kElementColumns; col_offset++) {
        destination[row_group * kElementColumns + col_offset] =
            static_cast<T>(source[row * stride_row + (col + col_offset) * stride_col]);
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
      typename OffsetCol = int>
  METAL_FUNC static constexpr void load_safe(
      thread fragment_vector_t<T>& destination,
      SourcePointerType source,
      StrideRow stride_row,
      StrideCol stride_col,
      RowLimit row_limit,
      ColumnLimit column_limit,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0) {
    const short2 simd_coordinate = get_coordinate();
    STEEL_PRAGMA_UNROLL
    for (short row_group = 0; row_group < kElementRows; row_group++) {
      const auto row = offset_row + row_group * kElementRowStride + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      STEEL_PRAGMA_UNROLL
      for (short col_offset = 0; col_offset < kElementColumns; col_offset++) {
        if (row < row_limit && (col + col_offset) < column_limit) {
          destination[row_group * kElementColumns + col_offset] =
              static_cast<T>(source[row * stride_row + (col + col_offset) * stride_col]);
        } else {
          destination[row_group * kElementColumns + col_offset] = T(0);
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
      typename OffsetCol = int>
  METAL_FUNC static constexpr void store(
      const thread fragment_vector_t<T>& source,
      DestinationPointerType destination,
      StrideRow stride_row,
      StrideCol stride_col,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0) {
    using OutputElementType = metal::pointer_element_t<DestinationPointerType>;
    const short2 simd_coordinate = get_coordinate();
    STEEL_PRAGMA_UNROLL
    for (short row_group = 0; row_group < kElementRows; row_group++) {
      const auto row = offset_row + row_group * kElementRowStride + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      STEEL_PRAGMA_UNROLL
      for (short col_offset = 0; col_offset < kElementColumns; col_offset++) {
        destination[row * stride_row + (col + col_offset) * stride_col] =
            static_cast<OutputElementType>(source[row_group * kElementColumns + col_offset]);
      }
    }
  }

  /// Stores a fragment register with bounds checking on both row and column dimensions.
  template <
      typename T,
      typename DestinationPointerType,
      typename StrideRow,
      typename StrideCol,
      typename RowLimit,
      typename ColumnLimit,
      typename OffsetRow = int,
      typename OffsetCol = int>
  METAL_FUNC static constexpr void store_safe(
      const thread fragment_vector_t<T>& source,
      DestinationPointerType destination,
      StrideRow stride_row,
      StrideCol stride_col,
      RowLimit row_limit,
      ColumnLimit column_limit,
      OffsetRow offset_row = 0,
      OffsetCol offset_col = 0) {
    using OutputElementType = metal::pointer_element_t<DestinationPointerType>;
    const short2 simd_coordinate = get_coordinate();
    STEEL_PRAGMA_UNROLL
    for (short row_group = 0; row_group < kElementRows; row_group++) {
      const auto row = offset_row + row_group * kElementRowStride + simd_coordinate.y;
      const auto col = offset_col + simd_coordinate.x;
      STEEL_PRAGMA_UNROLL
      for (short col_offset = 0; col_offset < kElementColumns; col_offset++) {
        if (row < row_limit && (col + col_offset) < column_limit) {
          destination[row * stride_row + (col + col_offset) * stride_col] =
              static_cast<OutputElementType>(source[row_group * kElementColumns + col_offset]);
        }
      }
    }
  }
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Section 2: Pre-M5 Cooperative Tensor GEMM
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Performs a complete GEMM for one output subtile using Metal's cooperative tensor API.
///
/// Loads A and B directly into cooperative tensors using get_multidimensional_index()
/// for correct coordinate mapping on all Apple Silicon generations (M1-M4).
/// Coordinates are precomputed once before the K loop for efficiency.
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
    bool transpose_right>
METAL_FUNC void cooperative_tensor_gemm(
    const device LeftInputType* left_input_pointer, int leading_dimension_a,
    const device RightInputType* right_input_pointer, int leading_dimension_b,
    device OutputType* output_pointer, int leading_dimension_output,
    int k_dimension,
    short row_limit, short column_limit) {

  constexpr auto matmul_descriptor = mpp::tensor_ops::matmul2d_descriptor(
      subtile_rows, subtile_columns, unroll_k,
      transpose_left, transpose_right, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  mpp::tensor_ops::matmul2d<matmul_descriptor, metal::execution_simdgroup> matmul_operation;

  auto left_tensor = matmul_operation.template get_left_input_cooperative_tensor<LeftInputType, RightInputType, AccumulatorType>();
  auto right_tensor = matmul_operation.template get_right_input_cooperative_tensor<LeftInputType, RightInputType, AccumulatorType>();
  auto accumulator_tensor = matmul_operation.template get_destination_cooperative_tensor<
      decltype(left_tensor), decltype(right_tensor), AccumulatorType>();

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
    const int per_lane = (left_capacity + right_capacity + accumulator_capacity) * 2 + 8 * 2;
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

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < left_capacity; i++) {
    auto coord = left_tensor.get_multidimensional_index(i);
    left_col[i] = coord[0];
    left_row[i] = coord[1];
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < right_capacity; i++) {
    auto coord = right_tensor.get_multidimensional_index(i);
    right_col[i] = coord[0];
    right_row[i] = coord[1];
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < accumulator_capacity; i++) {
    auto coord = accumulator_tensor.get_multidimensional_index(i);
    output_col[i] = coord[0];
    output_row[i] = coord[1];
    output_valid[i] = accumulator_tensor.is_valid_element(i);
  }

  int left_base_offset[16];
  int right_base_offset[16];
  bool left_in_bounds[16];
  bool right_in_bounds[16];

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < left_capacity; i++) {
    if constexpr (!transpose_left) {
      left_base_offset[i] = left_row[i] * leading_dimension_a;
      left_in_bounds[i] = left_row[i] < row_limit;
    } else {
      left_base_offset[i] = left_col[i] * leading_dimension_a;
      left_in_bounds[i] = left_col[i] < row_limit;
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < right_capacity; i++) {
    if constexpr (!transpose_right) {
      right_base_offset[i] = right_col[i];
      right_in_bounds[i] = right_col[i] < column_limit;
    } else {
      right_base_offset[i] = right_row[i] * leading_dimension_b;
      right_in_bounds[i] = right_row[i] < column_limit;
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < accumulator_capacity; i++) {
    accumulator_tensor[i] = AccumulatorType(0);
  }

  for (int k = 0; k < k_dimension; k += unroll_k) {
    const short k_remaining = short(min(int(unroll_k), k_dimension - k));

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < left_capacity; i++) {
      const short k_coordinate = transpose_left ? left_row[i] : left_col[i];
      if (left_in_bounds[i] && k_coordinate < k_remaining) {
        if constexpr (!transpose_left) {
          left_tensor[i] = left_input_pointer[left_base_offset[i] + left_col[i]];
        } else {
          left_tensor[i] = left_input_pointer[left_base_offset[i] + left_row[i]];
        }
      } else {
        left_tensor[i] = LeftInputType(0);
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < right_capacity; i++) {
      const short k_coordinate = transpose_right ? right_col[i] : right_row[i];
      if (right_in_bounds[i] && k_coordinate < k_remaining) {
        if constexpr (!transpose_right) {
          right_tensor[i] = right_input_pointer[right_row[i] * leading_dimension_b + right_col[i]];
        } else {
          right_tensor[i] = right_input_pointer[right_base_offset[i] + right_col[i]];
        }
      } else {
        right_tensor[i] = RightInputType(0);
      }
    }

    matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);

    left_input_pointer += transpose_left ? (unroll_k * leading_dimension_a) : unroll_k;
    right_input_pointer += transpose_right ? unroll_k : (unroll_k * leading_dimension_b);
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < accumulator_capacity; i++) {
    if (output_valid[i] && output_row[i] < row_limit && output_col[i] < column_limit) {
      output_pointer[output_row[i] * leading_dimension_output + output_col[i]] = OutputType(accumulator_tensor[i]);
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
    short kRows_,
    short kCols_,
    typename FragmentLayout = SimdgroupFragmentLayout>
struct MppSubTile {
  STEEL_CONST short kRows = kRows_;
  STEEL_CONST short kCols = kCols_;
  STEEL_CONST short kFragmentRows = FragmentLayout::kFragmentRows;
  STEEL_CONST short kFragmentColumns = FragmentLayout::kFragmentColumns;
  STEEL_CONST short kElementsPerFragment = FragmentLayout::kElementsPerFragment;
  STEEL_CONST short kSubTileRows = kRows / kFragmentRows;
  STEEL_CONST short kSubTileCols = kCols / kFragmentColumns;
  STEEL_CONST short kNumFragments = kSubTileRows * kSubTileCols;
  STEEL_CONST short kElementsPerSubTile = kNumFragments * kElementsPerFragment;
  STEEL_CONST int kRowsPerThread = kSubTileRows * FragmentLayout::kElementRows;
  STEEL_CONST int kColsPerThread = kSubTileCols * FragmentLayout::kElementColumns;
  STEEL_CONST short kFragmentThreadRows = FragmentLayout::kElementRows;
  STEEL_CONST short kFragmentThreadColumns = FragmentLayout::kElementColumns;
  STEEL_CONST short kFragmentRowStride = FragmentLayout::kElementRowStride;

  using fragment_type = typename FragmentLayout::template fragment_vector_t<T>;
  fragment_type value_fragments[kNumFragments];

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFragments; ++i) {
      value_fragments[i] = fragment_type(0);
    }
  }

  METAL_FUNC constexpr thread fragment_type& fragment_at(const short row, const short col) {
    return value_fragments[row * kSubTileCols + col];
  }

  METAL_FUNC constexpr const thread fragment_type& fragment_at(const short row, const short col) const {
    return value_fragments[row * kSubTileCols + col];
  }

  METAL_FUNC thread T* elements() { return reinterpret_cast<thread T*>(value_fragments); }
  METAL_FUNC const thread T* elements() const { return reinterpret_cast<const thread T*>(value_fragments); }

  template <typename SourcePointerType, typename StrideRow, typename StrideCol, typename OffsetRow = int, typename OffsetCol = int>
  METAL_FUNC constexpr void load(SourcePointerType source, StrideRow stride_row, StrideCol stride_col, OffsetRow offset_row = 0, OffsetCol offset_col = 0) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kSubTileCols; ++j) {
        FragmentLayout::load(fragment_at(i, j), source, stride_row, stride_col, offset_row + i * kFragmentRows, offset_col + j * kFragmentColumns);
      }
    }
  }

  template <typename DestinationPointerType, typename StrideRow, typename StrideCol, typename OffsetRow = int, typename OffsetCol = int>
  METAL_FUNC constexpr void store(DestinationPointerType destination, StrideRow stride_row, StrideCol stride_col, OffsetRow offset_row = 0, OffsetCol offset_col = 0) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kSubTileCols; ++j) {
        FragmentLayout::store(fragment_at(i, j), destination, stride_row, stride_col, offset_row + i * kFragmentRows, offset_col + j * kFragmentColumns);
      }
    }
  }

  template <typename SourcePointerType, typename StrideRow, typename StrideCol, typename RowLimit, typename ColumnLimit, typename OffsetRow = int, typename OffsetCol = int>
  METAL_FUNC constexpr void load_safe(SourcePointerType source, StrideRow stride_row, StrideCol stride_col, RowLimit row_limit, ColumnLimit column_limit, OffsetRow offset_row = 0, OffsetCol offset_col = 0) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kSubTileCols; ++j) {
        FragmentLayout::load_safe(fragment_at(i, j), source, stride_row, stride_col, row_limit, column_limit, offset_row + (i * kFragmentRows), offset_col + (j * kFragmentColumns));
      }
    }
  }

  template <typename DestinationPointerType, typename StrideRow, typename StrideCol, typename RowLimit, typename ColumnLimit, typename OffsetRow = int, typename OffsetCol = int>
  METAL_FUNC constexpr void store_safe(DestinationPointerType destination, StrideRow stride_row, StrideCol stride_col, RowLimit row_limit, ColumnLimit column_limit, OffsetRow offset_row = 0, OffsetCol offset_col = 0) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kSubTileCols; ++j) {
        FragmentLayout::store_safe(fragment_at(i, j), destination, stride_row, stride_col, row_limit, column_limit, offset_row + (i * kFragmentRows), offset_col + (j * kFragmentColumns));
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

  STEEL_PRAGMA_UNROLL
  for (short tile_m = 0; tile_m < tiles_m; tile_m++) {
    STEEL_PRAGMA_UNROLL
    for (short tile_k = 0; tile_k < tiles_k; tile_k++) {
      const short fragment_row = transpose_left ? tile_k : tile_m;
      const short fragment_col = transpose_left ? tile_m : tile_k;
      STEEL_PRAGMA_UNROLL
      for (short element = 0; element < 8; element++) {
        left_tensor[(tiles_k * tile_m + tile_k) * 8 + element] = left_input.fragment_at(fragment_row, fragment_col)[element];
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short tile_n = 0; tile_n < tiles_n; tile_n++) {
    STEEL_PRAGMA_UNROLL
    for (short tile_k = 0; tile_k < tiles_k; tile_k++) {
      const short fragment_row = transpose_right ? tile_n : tile_k;
      const short fragment_col = transpose_right ? tile_k : tile_n;
      STEEL_PRAGMA_UNROLL
      for (short element = 0; element < 8; element++) {
        right_tensor[(tiles_n * tile_k + tile_n) * 8 + element] = right_input.fragment_at(fragment_row, fragment_col)[element];
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < accumulator_tensor.get_capacity(); i++) {
    accumulator_tensor[i] = accumulator.elements()[i];
  }

  matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < accumulator_tensor.get_capacity(); i++) {
    accumulator.elements()[i] = accumulator_tensor[i];
  }
}

/// A grid of MppSubTile elements forming a larger tile.
template <typename T, short kTileRows_, short kTileCols_, class SubTileType>
struct MppTile {
  using subtile_type = SubTileType;
  using element_type = T;

  STEEL_CONST short kSubTileRows = subtile_type::kRows;
  STEEL_CONST short kSubTileCols = subtile_type::kCols;
  STEEL_CONST short kElementsPerSubTile = subtile_type::kElementsPerSubTile;
  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;
  STEEL_CONST short kRows = kTileRows * kSubTileRows;
  STEEL_CONST short kCols = kTileCols * kSubTileCols;
  STEEL_CONST short kSubTiles = kTileRows * kTileCols;
  STEEL_CONST short kElementsPerTile = kSubTiles * kElementsPerSubTile;
  STEEL_CONST short kRowsPerThread = kTileRows * subtile_type::kRowsPerThread;
  STEEL_CONST short kColsPerThread = kTileCols * subtile_type::kColsPerThread;

  subtile_type value_subtiles[kSubTiles];

  METAL_FUNC MppTile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTiles; ++i) { value_subtiles[i].clear(); }
  }

  METAL_FUNC constexpr thread subtile_type& subtile_at(const short row, const short col) {
    return value_subtiles[row * kTileCols + col];
  }

  METAL_FUNC constexpr const thread subtile_type& subtile_at(const short row, const short col) const {
    return value_subtiles[row * kTileCols + col];
  }

  METAL_FUNC thread element_type* elements() { return reinterpret_cast<thread element_type*>(value_subtiles[0].elements()); }
  METAL_FUNC const thread element_type* elements() const { return reinterpret_cast<const thread element_type*>(value_subtiles[0].elements()); }

  template <typename U>
  METAL_FUNC void load(const device U* source, const int leading_dimension) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).load(&source[(i * kSubTileRows * leading_dimension + j * kSubTileCols)], leading_dimension, 1);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store(device U* destination, const int leading_dimension) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).store(&destination[(i * kSubTileRows * leading_dimension + j * kSubTileCols)], leading_dimension, 1);
      }
    }
  }

  template <typename U>
  METAL_FUNC void load_safe(const device U* source, const int leading_dimension, const short2 tile_dimensions) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).load_safe(source, leading_dimension, 1, tile_dimensions.y, tile_dimensions.x, i * kSubTileRows, j * kSubTileCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store_safe(device U* destination, const int leading_dimension, const short2 tile_dimensions) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).store_safe(destination, leading_dimension, 1, tile_dimensions.y, tile_dimensions.x, i * kSubTileRows, j * kSubTileCols);
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

  constexpr short tiles_m = AccumulatorTile::kTileRows;
  constexpr short tiles_n = AccumulatorTile::kTileCols;
  constexpr short tiles_k = transpose_left ? LeftTile::kTileRows : LeftTile::kTileCols;

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < tiles_m; ++i) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < tiles_n; ++j) {
      STEEL_PRAGMA_UNROLL
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

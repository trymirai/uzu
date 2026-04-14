#pragma once

#include "simdgroup_fragment.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// ThreadgroupTile - manages the GEMM computation for a threadgroup
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename U,
    int BLOCK_ROWS,
    int BLOCK_COLS,
    int BLOCK_DEPTH,
    int SIMDGROUPS_PER_ROW,
    int SIMDGROUPS_PER_COLUMN,
    bool transpose_a,
    bool transpose_b,
    ushort THREADGROUP_LEADING_DIMENSION_A,
    ushort THREADGROUP_LEADING_DIMENSION_B,
    typename AccumulatorType = float,
    typename Epilogue = TransformNone<U, AccumulatorType>>
struct ThreadgroupTile {
  METAL_CONST ushort SIMDGROUP_BLOCK_SIZE = 8;
  using SimdgroupMultiplyAccumulateType = SimdgroupMultiplyAccumulate<
      AccumulatorType,
      SIMDGROUP_BLOCK_SIZE,
      SIMDGROUP_BLOCK_SIZE>;

  METAL_CONST ushort TILE_ROW_STRIDE = SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW;
  METAL_CONST ushort TILE_COL_STRIDE =
      SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN;

  METAL_CONST ushort TILE_ROWS =
      BLOCK_ROWS / (SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW);
  METAL_CONST ushort TILE_COLS =
      BLOCK_COLS / (SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN);

  METAL_CONST ushort A_STRIDE_ROW =
      transpose_a ? 1 : THREADGROUP_LEADING_DIMENSION_A;
  METAL_CONST ushort A_STRIDE_INNER =
      transpose_a ? THREADGROUP_LEADING_DIMENSION_A : 1;

  METAL_CONST ushort B_STRIDE_INNER =
      transpose_b ? 1 : THREADGROUP_LEADING_DIMENSION_B;
  METAL_CONST ushort B_STRIDE_COL =
      transpose_b ? THREADGROUP_LEADING_DIMENSION_B : 1;

  METAL_CONST ushort TILE_STRIDE_A = SIMDGROUP_BLOCK_SIZE * A_STRIDE_INNER;
  METAL_CONST ushort TILE_STRIDE_B = SIMDGROUP_BLOCK_SIZE * B_STRIDE_INNER;

  SimdgroupFragment<AccumulatorType, TILE_ROWS, 1, SimdgroupMultiplyAccumulateType>
      a_fragment;
  SimdgroupFragment<AccumulatorType, 1, TILE_COLS, SimdgroupMultiplyAccumulateType>
      b_fragment;
  SimdgroupFragment<
      AccumulatorType,
      TILE_ROWS,
      TILE_COLS,
      SimdgroupMultiplyAccumulateType>
      c_fragment;

  ushort simdgroup_row_offset;
  ushort simdgroup_col_offset;

  ushort a_shared_offset;
  ushort b_shared_offset;

  METAL_FUNC ThreadgroupTile(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  ) {
    ushort tile_row_base =
        SIMDGROUP_BLOCK_SIZE * (simd_group_id / SIMDGROUPS_PER_COLUMN);
    ushort tile_col_base =
        SIMDGROUP_BLOCK_SIZE * (simd_group_id % SIMDGROUPS_PER_COLUMN);

    ushort2 simdgroup_coordinates =
        SimdgroupMultiplyAccumulateType::get_lane_coordinates(simd_lane_id);
    simdgroup_row_offset = simdgroup_coordinates.y;
    simdgroup_col_offset = simdgroup_coordinates.x;

    a_shared_offset = (tile_row_base + simdgroup_row_offset) * A_STRIDE_ROW +
                      (simdgroup_col_offset)*A_STRIDE_INNER;
    b_shared_offset = (simdgroup_row_offset)*B_STRIDE_INNER +
                      (tile_col_base + simdgroup_col_offset) * B_STRIDE_COL;

    simdgroup_row_offset += tile_row_base;
    simdgroup_col_offset += tile_col_base;
  }

  METAL_FUNC void multiply_accumulate(
      const threadgroup T* a_shared,
      const threadgroup T* b_shared
  ) {
    a_shared += a_shared_offset;
    b_shared += b_shared_offset;

    METAL_PRAGMA_UNROLL
    for (ushort k_block_index = 0; k_block_index < BLOCK_DEPTH;
         k_block_index += SIMDGROUP_BLOCK_SIZE) {
      simdgroup_barrier(mem_flags::mem_none);

      a_fragment.template load<
          T,
          SIMDGROUPS_PER_ROW,
          1,
          A_STRIDE_ROW,
          A_STRIDE_INNER>(a_shared);

      simdgroup_barrier(mem_flags::mem_none);

      b_fragment.template load<
          T,
          1,
          SIMDGROUPS_PER_COLUMN,
          B_STRIDE_INNER,
          B_STRIDE_COL>(b_shared);

      simdgroup_barrier(mem_flags::mem_none);

      tile_multiply_accumulate(c_fragment, a_fragment, b_fragment, c_fragment);

      a_shared += TILE_STRIDE_A;
      b_shared += TILE_STRIDE_B;
    }
  }

  METAL_FUNC void store_result(device U* D, const int leading_dimension_d) {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < decltype(c_fragment)::ELEMENTS_PER_FRAGMENT; i++) {
      c_fragment.elements()[i] = Epilogue::apply(c_fragment.elements()[i]);
    }

    D += simdgroup_row_offset * leading_dimension_d + simdgroup_col_offset;

    c_fragment.template store<U, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN>(
        D,
        leading_dimension_d
    );
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int leading_dimension_d,
      short2 destination_tile_dimensions
  ) {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < decltype(c_fragment)::ELEMENTS_PER_FRAGMENT; i++) {
      c_fragment.elements()[i] = Epilogue::apply(c_fragment.elements()[i]);
    }

    D += simdgroup_row_offset * leading_dimension_d + simdgroup_col_offset;
    destination_tile_dimensions -=
        short2(simdgroup_col_offset, simdgroup_row_offset);

    if (destination_tile_dimensions.x <= 0 ||
        destination_tile_dimensions.y <= 0)
      return;

    c_fragment
        .template store_safe<U, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN>(
            D,
            leading_dimension_d,
            destination_tile_dimensions
        );
  }

  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int leading_dimension_c,
      const int column_stride_c,
      thread const EpilogueOp& epilogue_operation
  ) {
    const device U* c_pointer = C + simdgroup_row_offset * leading_dimension_c +
                                simdgroup_col_offset * column_stride_c;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < TILE_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < TILE_COLS; j++) {
        thread auto& block_data = c_fragment.multiply_accumulate_at(i, j);
        METAL_PRAGMA_UNROLL
        for (ushort k = 0;
             k < decltype(c_fragment)::ELEMENTS_PER_MULTIPLY_ACCUMULATE;
             k++) {
          const ushort row_offset =
              i * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW;
          const ushort col_offset =
              j * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN;
          U c_value = c_pointer
              [row_offset * leading_dimension_c + col_offset * column_stride_c +
               k];
          block_data[k] = epilogue_operation.apply(
              block_data[k],
              static_cast<AccumulatorType>(c_value)
          );
        }
      }
    }
  }

  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int leading_dimension_c,
      const int column_stride_c,
      short2 tile_dimensions,
      thread const EpilogueOp& epilogue_operation
  ) {
    const device U* c_pointer = C + simdgroup_row_offset * leading_dimension_c +
                                simdgroup_col_offset * column_stride_c;
    tile_dimensions -= short2(simdgroup_col_offset, simdgroup_row_offset);

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < TILE_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < TILE_COLS; j++) {
        thread auto& block_data = c_fragment.multiply_accumulate_at(i, j);
        const ushort row_offset =
            i * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW;
        const ushort col_offset =
            j * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN;
        METAL_PRAGMA_UNROLL
        for (ushort k = 0;
             k < decltype(c_fragment)::ELEMENTS_PER_MULTIPLY_ACCUMULATE;
             k++) {
          if (row_offset < tile_dimensions.y &&
              col_offset + k < tile_dimensions.x) {
            U c_value = c_pointer
                [row_offset * leading_dimension_c +
                 col_offset * column_stride_c + k];
            block_data[k] = epilogue_operation.apply(
                block_data[k],
                static_cast<AccumulatorType>(c_value)
            );
          }
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu

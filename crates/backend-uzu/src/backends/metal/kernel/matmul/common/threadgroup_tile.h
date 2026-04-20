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
  using SimdgroupFragmentOpsType = SimdgroupFragmentOps<AccumulatorType>;

  METAL_CONST ushort TILE_ROW_STRIDE =
      SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW;
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

  Fragment<AccumulatorType, TILE_ROWS, 1, SimdgroupFragmentOpsType> a_fragment;
  Fragment<AccumulatorType, 1, TILE_COLS, SimdgroupFragmentOpsType> b_fragment;
  Fragment<AccumulatorType, TILE_ROWS, TILE_COLS, SimdgroupFragmentOpsType>
      c_fragment;

  ushort tile_row_offset;
  ushort tile_col_offset;

  ushort a_shared_offset;
  ushort b_shared_offset;

  METAL_FUNC ThreadgroupTile(const thread ThreadContext& thread_context)
      : a_fragment(thread_context),
        b_fragment(thread_context),
        c_fragment(thread_context) {
    const ushort simd_group_id = ushort(thread_context.threadgroup_index);
    tile_row_offset =
        SIMDGROUP_BLOCK_SIZE * (simd_group_id / SIMDGROUPS_PER_COLUMN);
    tile_col_offset =
        SIMDGROUP_BLOCK_SIZE * (simd_group_id % SIMDGROUPS_PER_COLUMN);

    a_shared_offset = tile_row_offset * A_STRIDE_ROW;
    b_shared_offset = tile_col_offset * B_STRIDE_COL;
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

      a_fragment.load(
          a_shared,
          A_STRIDE_ROW,
          A_STRIDE_INNER,
          SIMDGROUPS_PER_ROW,
          1
      );

      simdgroup_barrier(mem_flags::mem_none);

      b_fragment.load(
          b_shared,
          B_STRIDE_INNER,
          B_STRIDE_COL,
          1,
          SIMDGROUPS_PER_COLUMN
      );

      simdgroup_barrier(mem_flags::mem_none);

      SimdgroupFragmentOpsType::template tile_matmul<transpose_a, transpose_b>(
          c_fragment,
          a_fragment,
          b_fragment
      );

      a_shared += TILE_STRIDE_A;
      b_shared += TILE_STRIDE_B;
    }
  }

  METAL_FUNC void store_result(device U* D, const int leading_dimension_d) {
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < decltype(c_fragment)::ELEMENTS_PER_TILE;
         index++) {
      c_fragment.elements()[index] =
          Epilogue::apply(c_fragment.elements()[index]);
    }

    D += tile_row_offset * leading_dimension_d + tile_col_offset;

    c_fragment.store(
        D,
        leading_dimension_d,
        1,
        SIMDGROUPS_PER_ROW,
        SIMDGROUPS_PER_COLUMN
    );
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int leading_dimension_d,
      short2 destination_tile_dimensions
  ) {
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < decltype(c_fragment)::ELEMENTS_PER_TILE;
         index++) {
      c_fragment.elements()[index] =
          Epilogue::apply(c_fragment.elements()[index]);
    }

    D += tile_row_offset * leading_dimension_d + tile_col_offset;
    destination_tile_dimensions -= short2(tile_col_offset, tile_row_offset);

    if (destination_tile_dimensions.x <= 0 ||
        destination_tile_dimensions.y <= 0)
      return;

    c_fragment.store_safe(
        D,
        leading_dimension_d,
        1,
        destination_tile_dimensions.y,
        destination_tile_dimensions.x,
        SIMDGROUPS_PER_ROW,
        SIMDGROUPS_PER_COLUMN
    );
  }

  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int leading_dimension_c,
      const int column_stride_c,
      thread const EpilogueOp& epilogue_operation
  ) {
    const device U* c_pointer =
        C + tile_row_offset * leading_dimension_c + tile_col_offset * column_stride_c;
    const short2 position =
        SimdgroupFragmentOpsType::get_position(c_fragment.thread_context);

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < TILE_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < TILE_COLS; j++) {
        thread auto& block_data = c_fragment.fragment_at(i, j);
        const ushort row_base =
            i * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW + ushort(position.y);
        const ushort col_base =
            j * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN +
            ushort(position.x);
        METAL_PRAGMA_UNROLL
        for (ushort row_offset = 0;
             row_offset < SimdgroupFragmentOpsType::THREAD_ELEMENT_ROWS;
             row_offset++) {
          METAL_PRAGMA_UNROLL
          for (ushort col_offset = 0;
               col_offset < SimdgroupFragmentOpsType::THREAD_ELEMENT_COLS;
               col_offset++) {
            const ushort element_index =
                row_offset * SimdgroupFragmentOpsType::THREAD_ELEMENT_COLS +
                col_offset;
            const U c_value =
                c_pointer[(row_base + row_offset) * leading_dimension_c +
                          (col_base + col_offset) * column_stride_c];
            block_data[element_index] = epilogue_operation.apply(
                block_data[element_index],
                static_cast<AccumulatorType>(c_value)
            );
          }
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
    const device U* c_pointer =
        C + tile_row_offset * leading_dimension_c + tile_col_offset * column_stride_c;
    tile_dimensions -= short2(tile_col_offset, tile_row_offset);
    const short2 position =
        SimdgroupFragmentOpsType::get_position(c_fragment.thread_context);

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < TILE_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < TILE_COLS; j++) {
        thread auto& block_data = c_fragment.fragment_at(i, j);
        const ushort row_base =
            i * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW + ushort(position.y);
        const ushort col_base =
            j * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN +
            ushort(position.x);
        METAL_PRAGMA_UNROLL
        for (ushort row_offset = 0;
             row_offset < SimdgroupFragmentOpsType::THREAD_ELEMENT_ROWS;
             row_offset++) {
          METAL_PRAGMA_UNROLL
          for (ushort col_offset = 0;
               col_offset < SimdgroupFragmentOpsType::THREAD_ELEMENT_COLS;
               col_offset++) {
            if ((row_base + row_offset) < tile_dimensions.y &&
                (col_base + col_offset) < tile_dimensions.x) {
              const ushort element_index =
                  row_offset * SimdgroupFragmentOpsType::THREAD_ELEMENT_COLS +
                  col_offset;
              const U c_value =
                  c_pointer[(row_base + row_offset) * leading_dimension_c +
                            (col_base + col_offset) * column_stride_c];
              block_data[element_index] = epilogue_operation.apply(
                  block_data[element_index],
                  static_cast<AccumulatorType>(c_value)
              );
            }
          }
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu

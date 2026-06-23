#pragma once

#include "../../common/thread_context.h"
#include "fragment.h"
#include "simdgroup_fragment_ops.h"

using namespace metal;

namespace uzu {
namespace matmul {

template <
    typename AT,
    typename BT,
    typename DT,
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
    typename Epilogue = TransformNone<DT, AccumulatorType>>
struct ThreadgroupTile {
  METAL_CONST ushort SIMDGROUP_BLOCK_SIZE = 8;
  using SimdgroupMultiplyAccumulateType =
      SimdgroupMultiplyAccumulate<AccumulatorType, SIMDGROUP_BLOCK_SIZE, SIMDGROUP_BLOCK_SIZE>;

  METAL_CONST ushort TILE_ROW_STRIDE = SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW;
  METAL_CONST ushort TILE_COL_STRIDE = SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN;

  METAL_CONST ushort TILE_ROWS = BLOCK_ROWS / (SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW);
  METAL_CONST ushort TILE_COLS = BLOCK_COLS / (SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN);

  METAL_CONST ushort A_STRIDE_ROW = transpose_a ? 1 : THREADGROUP_LEADING_DIMENSION_A;
  METAL_CONST ushort A_STRIDE_INNER = transpose_a ? THREADGROUP_LEADING_DIMENSION_A : 1;

  METAL_CONST ushort B_STRIDE_INNER = transpose_b ? 1 : THREADGROUP_LEADING_DIMENSION_B;
  METAL_CONST ushort B_STRIDE_COL = transpose_b ? THREADGROUP_LEADING_DIMENSION_B : 1;

  METAL_CONST ushort TILE_STRIDE_A = SIMDGROUP_BLOCK_SIZE * A_STRIDE_INNER;
  METAL_CONST ushort TILE_STRIDE_B = SIMDGROUP_BLOCK_SIZE * B_STRIDE_INNER;

  Fragment<AccumulatorType, TILE_ROWS, 1, SimdgroupFragmentOps> a_fragment;
  Fragment<AccumulatorType, 1, TILE_COLS, SimdgroupFragmentOps> b_fragment;
  Fragment<AccumulatorType, TILE_ROWS, TILE_COLS, SimdgroupFragmentOps> c_fragment;

  ushort simdgroup_row_offset;
  ushort simdgroup_col_offset;

  ushort a_shared_offset;
  ushort b_shared_offset;

  METAL_FUNC ThreadgroupTile(const thread ThreadContext& thread_context) {
    ushort tile_row_base = SIMDGROUP_BLOCK_SIZE * (thread_context.simdgroup_index / SIMDGROUPS_PER_COLUMN);
    ushort tile_col_base = SIMDGROUP_BLOCK_SIZE * (thread_context.simdgroup_index % SIMDGROUPS_PER_COLUMN);

    const ushort2 simdgroup_coordinates =
        ushort2(SimdgroupMultiplyAccumulateType::get_lane_coordinates(thread_context.simd_lane_id));
    simdgroup_row_offset = simdgroup_coordinates.y;
    simdgroup_col_offset = simdgroup_coordinates.x;

    a_shared_offset = (tile_row_base + simdgroup_row_offset) * A_STRIDE_ROW + (simdgroup_col_offset)*A_STRIDE_INNER;
    b_shared_offset = (simdgroup_row_offset)*B_STRIDE_INNER + (tile_col_base + simdgroup_col_offset) * B_STRIDE_COL;

    simdgroup_row_offset += tile_row_base;
    simdgroup_col_offset += tile_col_base;
  }

  METAL_FUNC void multiply_accumulate(const threadgroup AT* a_shared, const threadgroup BT* b_shared) {
    a_shared += a_shared_offset;
    b_shared += b_shared_offset;

    METAL_PRAGMA_UNROLL
    for (ushort k_block_index = 0; k_block_index < BLOCK_DEPTH; k_block_index += SIMDGROUP_BLOCK_SIZE) {
      fragment_load<AT, SIMDGROUPS_PER_ROW, 1, A_STRIDE_ROW, A_STRIDE_INNER>(a_fragment, a_shared);

      fragment_load<BT, 1, SIMDGROUPS_PER_COLUMN, B_STRIDE_INNER, B_STRIDE_COL>(b_fragment, b_shared);

      fragment_matmul(c_fragment, a_fragment, b_fragment);

      a_shared += TILE_STRIDE_A;
      b_shared += TILE_STRIDE_B;
    }
  }

  METAL_FUNC void store_result(device DT* D, const int leading_dimension_d) {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < decltype(c_fragment)::ELEMENTS_PER_FRAGMENT; i++) {
      c_fragment.elements()[i] = Epilogue::apply(c_fragment.elements()[i]);
    }

    D += simdgroup_row_offset * leading_dimension_d + simdgroup_col_offset;

    fragment_store<DT, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN>(c_fragment, D, leading_dimension_d);
  }

  METAL_FUNC void store_result_safe(device DT* D, const int leading_dimension_d, short2 destination_tile_dimensions) {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < decltype(c_fragment)::ELEMENTS_PER_FRAGMENT; i++) {
      c_fragment.elements()[i] = Epilogue::apply(c_fragment.elements()[i]);
    }

    D += simdgroup_row_offset * leading_dimension_d + simdgroup_col_offset;
    destination_tile_dimensions -= short2(simdgroup_col_offset, simdgroup_row_offset);

    if (destination_tile_dimensions.x <= 0 || destination_tile_dimensions.y <= 0)
      return;

    fragment_store_safe<DT, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN>(
        c_fragment,
        D,
        leading_dimension_d,
        destination_tile_dimensions
    );
  }

  template <class Fn>
  METAL_FUNC void for_each_output(Fn fn) {
    thread AccumulatorType* data = c_fragment.elements();
    constexpr ushort epm = SimdgroupFragmentOps::ELEMENTS_PER_THREAD;
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < TILE_ROWS; i++) {
      const ushort row_offset = i * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW;
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < TILE_COLS; j++) {
        const ushort col_offset = j * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN;
        METAL_PRAGMA_UNROLL
        for (ushort k = 0; k < epm; k++) {
          fn(row_offset, col_offset, k, data[(i * TILE_COLS + j) * epm + k]);
        }
      }
    }
  }

  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue(const device DT* C, const int ld_c, const int cstride_c, thread const EpilogueOp& op) {
    const device DT* c_ptr = C + simdgroup_row_offset * ld_c + simdgroup_col_offset * cstride_c;
    for_each_output([&](ushort row_offset, ushort col_offset, ushort k, thread AccumulatorType& v) {
      v = op.apply(v, static_cast<AccumulatorType>(c_ptr[row_offset * ld_c + col_offset * cstride_c + k]));
    });
  }

  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue_safe(
      const device DT* C,
      const int ld_c,
      const int cstride_c,
      short2 tile_dimensions,
      thread const EpilogueOp& op
  ) {
    const device DT* c_ptr = C + simdgroup_row_offset * ld_c + simdgroup_col_offset * cstride_c;
    tile_dimensions -= short2(simdgroup_col_offset, simdgroup_row_offset);
    for_each_output([&](ushort row_offset, ushort col_offset, ushort k, thread AccumulatorType& v) {
      if (row_offset < tile_dimensions.y && col_offset + k < tile_dimensions.x)
        v = op.apply(v, static_cast<AccumulatorType>(c_ptr[row_offset * ld_c + col_offset * cstride_c + k]));
    });
  }

  METAL_FUNC void apply_bias(const device BT* bias) {
    const device BT* bias_ptr = bias + simdgroup_col_offset;
    for_each_output([&](ushort, ushort col_offset, ushort k, thread AccumulatorType& v) {
      v += static_cast<AccumulatorType>(bias_ptr[col_offset + k]);
    });
  }

  METAL_FUNC void apply_bias_safe(const device BT* bias, short2 tile_dimensions) {
    const device BT* bias_ptr = bias + simdgroup_col_offset;
    tile_dimensions -= short2(simdgroup_col_offset, simdgroup_row_offset);
    for_each_output([&](ushort row_offset, ushort col_offset, ushort k, thread AccumulatorType& v) {
      if (row_offset < tile_dimensions.y && col_offset + k < tile_dimensions.x)
        v += static_cast<AccumulatorType>(bias_ptr[col_offset + k]);
    });
  }
};

} // namespace matmul
} // namespace uzu

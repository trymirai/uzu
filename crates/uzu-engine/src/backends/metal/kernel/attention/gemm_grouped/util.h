#pragma once

#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../matmul/common/mxu_fragment/ops.h"

using namespace metal;

#define ATTENTION_GQA_SIMDGROUPS 8

template <uint BK, uint BD, uint D_SLICES>
struct AttentionGemmGroupedLayout {
  using Ops = uzu::matmul::MxuFragmentOps<>;

  UZU_CONST int SIMDGROUPS = ATTENTION_GQA_SIMDGROUPS;
  UZU_CONST int TILE_ROWS = Ops::FRAGMENT_ROWS;
  UZU_CONST int SLICE_COLS = int(BD) / int(D_SLICES);
  UZU_CONST int BANDS = SIMDGROUPS / int(D_SLICES);
  UZU_CONST int BLOCK_ROWS = BANDS * TILE_ROWS;
  UZU_CONST int SCORE_TILE_ELEMENTS = TILE_ROWS * int(BK);
  UZU_CONST int SCORE_EXCHANGE_SIZE = SIMDGROUPS * SCORE_TILE_ELEMENTS;
};

// `fn(row_in_lane, column, value)` over one lane's elements of `tile`.
template <class FragmentType, class Fn>
METAL_FUNC void map_lane_rows(thread FragmentType& tile, const short2 position, Fn fn) {
  using Ops = typename FragmentType::FragmentOpsType;
  using Element = typename FragmentType::ElementType;
  static_assert(FragmentType::ROW_FRAGMENTS == 1, "requires one row fragment");
  static_assert(
      Ops::THREAD_ELEMENT_ROWS * Ops::THREAD_ELEMENT_COLS == Ops::ELEMENTS_PER_THREAD,
      "requires row-major thread elements"
  );

  thread Element* data = tile.elements();
  METAL_PRAGMA_UNROLL
  for (ushort fragment_col = 0; fragment_col < FragmentType::COL_FRAGMENTS; ++fragment_col) {
    const ushort fragment_base = fragment_col * Ops::ELEMENTS_PER_THREAD;
    const short col_base = position.x + short(fragment_col) * short(Ops::FRAGMENT_COLS);
    METAL_PRAGMA_UNROLL
    for (ushort row = 0; row < Ops::THREAD_ELEMENT_ROWS; ++row) {
      METAL_PRAGMA_UNROLL
      for (ushort col = 0; col < Ops::THREAD_ELEMENT_COLS; ++col) {
        const ushort element = fragment_base + row * Ops::THREAD_ELEMENT_COLS + col;
        data[element] = Element(fn(row, col_base + short(col), data[element]));
      }
    }
  }
}

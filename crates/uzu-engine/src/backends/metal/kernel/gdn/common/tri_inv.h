#pragma once

#include "../../common/defines.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"
#include <metal_stdlib>

using namespace metal;
using namespace uzu::matmul;

template <uint BLOCK, uint A_STRIDE = BLOCK, typename OutPtr>
METAL_FUNC void invert_lower_triangular_block(
    OutPtr a_inv_block,
    threadgroup const float* a_tile,
    const uint block_size,
    const uint lane
) {
  if (lane >= BLOCK) {
    return;
  }

  const uint col = lane;
  float inverse_col[BLOCK] = {};
  inverse_col[col] = 1.0f;

  METAL_PRAGMA_UNROLL
  for (uint row = 0; row < BLOCK; row++) {
    if (row > col && row < block_size) {
      float acc = 0.0f;
      METAL_PRAGMA_UNROLL
      for (uint prev_row = 0; prev_row < BLOCK; prev_row++) {
        if (prev_row < row) {
          acc += a_tile[row * A_STRIDE + prev_row] * inverse_col[prev_row];
        }
      }
      inverse_col[row] = -acc;
    }
  }

  METAL_PRAGMA_UNROLL
  for (uint row = 0; row < BLOCK; row++) {
    a_inv_block[row * BLOCK + col] = inverse_col[row];
  }
}

template <uint BLOCK, uint NB, typename CarryT, typename ARow, typename ABlock, typename StoreBlock>
METAL_FUNC void solve_lower_triangular_slice(
    ARow a_row,
    ABlock a_block,
    const uint a_stride,
    const threadgroup float* diag_inverses,
    StoreBlock store_block,
    const uint slice,
    const uint lane
) {
  using Ops = SimdgroupFragmentOps;
  constexpr ushort ROW_FRAGMENTS = BLOCK / Ops::FRAGMENT_ROWS;
  using TileFragment = Fragment<float, ROW_FRAGMENTS, ROW_FRAGMENTS, Ops>;
  using MatrixFragment = OperandFragment<float, ROW_FRAGMENTS, ROW_FRAGMENTS, Ops>;
  using CarryFragment = OperandFragment<CarryT, ROW_FRAGMENTS, ROW_FRAGMENTS, Ops>;

  CarryFragment t_blocks[NB];
  METAL_PRAGMA_UNROLL
  for (uint b = 0; b < NB; ++b) {
    TileFragment t_block;
    t_block.clear();
    if (b >= slice) {
      const threadgroup float* b_tiles = a_row(b);
      TileFragment acc_t;
      acc_t.map_coords(lane, [&](short local_row, short local_col, float) UZU_ALWAYS_INLINE {
        return b * BLOCK + uint(local_row) == slice * BLOCK + uint(local_col) ? 1.0f : 0.0f;
      });
      METAL_PRAGMA_UNROLL
      for (uint m = 0; m < b; ++m) {
        if (m >= slice) {
          MatrixFragment a_frag;
          a_frag.load_from(lane, fragment_source(a_block(b_tiles, m), a_stride));
          a_frag.map([](float value) UZU_ALWAYS_INLINE { return -value; });
          fragment_mma(acc_t, a_frag, t_blocks[m]);
        }
      }
      MatrixFragment inv_frag;
      inv_frag.load_from(lane, fragment_source(diag_inverses + b * (BLOCK * BLOCK), BLOCK));
      fragment_mma(t_block, inv_frag, acc_t);
    }
    store_block(b, t_block);
    METAL_PRAGMA_UNROLL
    for (ushort e = 0; e < CarryFragment::ELEMENTS_PER_FRAGMENT; ++e) {
      t_blocks[b].elements()[e] = CarryT(t_block.elements()[e]);
    }
  }
}

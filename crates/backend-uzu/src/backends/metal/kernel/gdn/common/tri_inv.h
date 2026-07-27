#pragma once

#include "../../common/defines.h"
#include <metal_stdlib>

using namespace metal;

template <uint BLOCK>
METAL_FUNC void invert_lower_triangular_block(
    device float* a_inv_block,
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
          acc += a_tile[row * BLOCK + prev_row] * inverse_col[prev_row];
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

#pragma once

#include <metal_stdlib>
#include "../common/defines.h"

METAL_FUNC float rope_rotate(
    float element,
    float paired,
    const device float* inverse_frequencies,
    float scale,
    uint32_t position,
    uint32_t channel,
    uint32_t rope_dim
) {
  const uint32_t half_rope_dim = rope_dim / 2;
  const float angle = float(position) * inverse_frequencies[channel % half_rope_dim];
  float cos_val;
  const float sin_val = metal::precise::sincos(angle, cos_val);
  const float signed_paired = channel < half_rope_dim ? -paired : paired;
  return element * (cos_val * scale) + signed_paired * (sin_val * scale);
}

// The M-RoPE axis offset of a pair. Interleaved axes: pairs 1, 4, 7, ... add the grid row, pairs 2, 5, 8, ... the grid
// column, while the axis has pairs left. Otherwise the pairs of an axis are contiguous: position, then row, then
// column.
METAL_FUNC uint32_t mrope_axis_offset(
    uint32_t pair,
    uint32_t pairs,
    uint32_t grid_row,
    uint32_t grid_column,
    uint32_t height_pairs,
    uint32_t width_pairs,
    bool interleaved_axes
) {
  if (!interleaved_axes) {
    const uint32_t position_pairs = pairs - height_pairs - width_pairs;
    if (pair < position_pairs) {
      return 0;
    }
    return pair < position_pairs + height_pairs ? grid_row : grid_column;
  }
  if (pair % 3 == 1 && pair < 3 * height_pairs) {
    return grid_row;
  }
  if (pair % 3 == 2 && pair < 3 * width_pairs) {
    return grid_column;
  }
  return 0;
}

template <typename ResultT, typename ElementT>
METAL_FUNC ResultT apply_rope(
    const device ElementT* head,
    const device float* inverse_frequencies,
    float scale,
    uint32_t position,
    uint32_t head_dim_idx,
    uint32_t rope_dim
) {
  const uint32_t half_rope_dim = rope_dim / 2;
  const uint32_t paired_idx =
      head_dim_idx < half_rope_dim ? head_dim_idx + half_rope_dim : head_dim_idx - half_rope_dim;
  return static_cast<ResultT>(rope_rotate(
      float(head[head_dim_idx]),
      float(head[paired_idx]),
      inverse_frequencies,
      scale,
      position,
      head_dim_idx,
      rope_dim
  ));
}

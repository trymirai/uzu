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

// TODO: very ugly and wasteful, clean up and optimize
template <typename ElementT>
METAL_FUNC ElementT apply_rope(
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
  return static_cast<ElementT>(rope_rotate(
      float(head[head_dim_idx]),
      float(head[paired_idx]),
      inverse_frequencies,
      scale,
      position,
      head_dim_idx,
      rope_dim
  ));
}

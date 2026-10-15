#pragma once

#include <metal_stdlib>
#include "../common/defines.h"

// TODO: very ugly and wasteful, clean up and optimize
template <typename ElementT, typename RopeT>
METAL_FUNC ElementT apply_rope(
    const device ElementT* head,
    const device RopeT* cosines,
    const device RopeT* sines,
    uint32_t batch_idx,
    uint32_t head_dim_idx,
    uint32_t rope_dim
) {
  const uint32_t half_rope_dim = rope_dim / 2;
  const uint32_t paired_idx =
      head_dim_idx < half_rope_dim ? head_dim_idx + half_rope_dim : head_dim_idx - half_rope_dim;
  const float input = float(head[head_dim_idx]);
  const float paired = float(head[paired_idx]);
  const float signed_paired = head_dim_idx < half_rope_dim ? -paired : paired;
  const float cos_val = float(cosines[batch_idx * rope_dim + head_dim_idx]);
  const float sin_val = float(sines[batch_idx * rope_dim + head_dim_idx]);

  return static_cast<ElementT>(input * cos_val + signed_paired * sin_val);
}

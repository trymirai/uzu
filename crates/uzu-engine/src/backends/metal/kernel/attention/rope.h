#pragma once

#include <metal_stdlib>

using namespace metal;

template <typename T>
METAL_FUNC T apply_attention_rope(
    T input,
    const device T* head,
    const device float* cosines,
    const device float* sines,
    uint position,
    uint head_dim_idx,
    uint rope_dim
) {
  if (head_dim_idx >= rope_dim) {
    return input;
  }

  const uint half_rope_dim = rope_dim / 2;
  const bool first_half = head_dim_idx < half_rope_dim;
  const uint paired_idx = first_half ? head_dim_idx + half_rope_dim : head_dim_idx - half_rope_dim;
  const float paired = static_cast<float>(head[paired_idx]);
  const float signed_paired = first_half ? -paired : paired;

  const uint64_t coefficient_idx = static_cast<uint64_t>(position) * rope_dim + head_dim_idx;
  const float value = static_cast<float>(input) * cosines[coefficient_idx] + signed_paired * sines[coefficient_idx]
  return static_cast<T>(value);
}

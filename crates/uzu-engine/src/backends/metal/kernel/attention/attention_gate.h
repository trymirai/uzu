#pragma once

#include <metal_stdlib>

template <typename T>
METAL_FUNC T attention_apply_gate(const T result, const T gate) {
  const float sigmoid = 1.0f / (1.0f + metal::exp(-float(gate)));
  return T(float(result) * sigmoid);
}

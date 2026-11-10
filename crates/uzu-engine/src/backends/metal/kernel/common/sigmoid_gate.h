#pragma once

#include <metal_stdlib>

#include "defines.h"

namespace uzu {

template <typename T>
METAL_FUNC T apply_sigmoid_gate(const T value, const T gate) {
  const float sigmoid = 1.0f / (1.0f + metal::exp(-float(gate)));
  return T(float(value) * sigmoid);
}

} // namespace uzu

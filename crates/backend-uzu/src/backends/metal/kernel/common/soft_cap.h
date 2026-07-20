#pragma once

#include <metal_stdlib>

#include "defines.h"

namespace uzu {

template <typename T>
METAL_FUNC T apply_soft_cap(T value, float cap) {
  return static_cast<T>(cap * metal::fast::tanh(static_cast<float>(value) / cap));
}

} // namespace uzu

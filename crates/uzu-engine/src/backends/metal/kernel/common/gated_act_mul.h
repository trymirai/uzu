#pragma once

#include <metal_stdlib>

#include "../activation/activations.h"

using namespace metal;
using namespace uzu::activation_type;

template <typename T>
static METAL_FUNC float gated_act_mul(T value, T gate, ActivationType act_type, float alpha) {
  const T result = value * activate(gate, act_type, alpha);
  return static_cast<float>(result);
}

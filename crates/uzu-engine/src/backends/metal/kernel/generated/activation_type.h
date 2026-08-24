// Auto-generated from gpu_types/activation_type - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::activation_type {
static constant constexpr float GELU_K0 = 0.044715;

static constant constexpr float GELU_K1 = 0.7978846;

enum class ActivationType : uint32_t {
  SILU = 0,
  GELUApprox = 1,
  GELUExact = 2,
  IDENTITY = 3,
  SOFTPLUS = 4,
};
} // namespace uzu::activation_type

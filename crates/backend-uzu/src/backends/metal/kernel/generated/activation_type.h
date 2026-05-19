// Auto-generated from gpu_types/activation_type - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::activation_type {
enum class ActivationType : uint32_t {
  SILU = 0,
  GELUApprox = 1,
  GELUExact = 2,
  TANH = 3,
  IDENTITY = 4,
  SOFTPLUS = 5,
};
} // namespace uzu::activation_type

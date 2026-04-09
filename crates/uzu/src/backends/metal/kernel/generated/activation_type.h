// Auto-generated from gpu_types/activation_type.rs - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::activation_type {
enum class ActivationType : uint32_t {
  SILU = 0,
  GELU = 1,
  TANH = 2,
  IDENTITY = 3,
  SOFTPLUS = 4,
};
} // namespace uzu::activation_type

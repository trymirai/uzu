// Auto-generated from gpu_types/activation_quant - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::activation_quant {
enum class GemmActivationQuant : uint32_t {
  Disabled = 0,
  Int8Symmetric = 1,
  Int8Asymmetric = 2,
};

enum class ActivationScaleStat : uint32_t {
  AbsMax = 0,
  Rms = 1,
};

enum class ActivationScaleGranularity : uint32_t {
  GroupWise = 0,
  TokenWise = 1,
};
} // namespace uzu::activation_quant

// Auto-generated from gpu_types/quantization_method - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::quantization_method {
enum class QuantizationMethod : uint32_t {
  ScaleBias = 0,
  ScaleZeroPoint = 1,
  ScaleSymmetric = 2,
  Codebook = 3,
};
} // namespace uzu::quantization_method

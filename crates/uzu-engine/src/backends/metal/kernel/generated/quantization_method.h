// Auto-generated from gpu_types/quantization_method - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::quantization_method {
enum class QuantizationMethod : uint32_t {
  ScaleZeroPoint = 0,
  ScaleSymmetric = 1,
};
} // namespace uzu::quantization_method

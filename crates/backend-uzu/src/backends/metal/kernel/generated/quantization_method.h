// Auto-generated from gpu_types/quantization_method.rs - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::quantization_method {
enum class QuantizationMethod : uint32_t {
  ScaleBias = 0,
  ScaleZeroPoint = 1,
};
} // namespace uzu::quantization_method

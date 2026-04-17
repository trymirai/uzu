// Auto-generated from gpu_types/quantization.rs - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::quantization {
enum class QuantizationMode : uint32_t {
  UINT4 = 0,
  INT8 = 1,
  UINT8 = 2,
};
} // namespace uzu::quantization

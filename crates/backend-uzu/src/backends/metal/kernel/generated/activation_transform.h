// Auto-generated from gpu_types/activation_transform - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::activation_transform {
struct ActivationTransformOp {
  uint32_t raw_value;
  constexpr ActivationTransformOp() thread : raw_value(0) {}
  constexpr ActivationTransformOp(uint32_t __dsl_v) thread : raw_value(__dsl_v) {}
  static constant constexpr uint32_t INPUT_RHT = 1 << 0;
  static constant constexpr uint32_t OUTPUT_RHT = 1 << 1;
  static constant constexpr uint32_t QUANTIZE = 1 << 2;
  static constant constexpr uint32_t GROUP_SUMS = 1 << 3;
  constexpr bool contains(uint32_t flag) const thread { return (raw_value & flag) != 0; }
  constexpr bool contains(uint32_t flag) const constant { return (raw_value & flag) != 0; }
  constexpr uint32_t bits() const thread { return raw_value; }
  constexpr uint32_t bits() const constant { return raw_value; }
};
} // namespace uzu::activation_transform

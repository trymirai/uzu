// Auto-generated from gpu_types/activation_prepare - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::activation_prepare {
static constant constexpr float INT8_SYMMETRIC_QUANTIZATION_MAXIMUM = 127.0;

struct ActivationPrepareOps {
  uint raw_value;
  constexpr ActivationPrepareOps() thread : raw_value(0) {}
  constexpr ActivationPrepareOps(uint __dsl_v) thread : raw_value(__dsl_v) {}
  static constant constexpr uint INPUT_RHT = 1 << 0;
  static constant constexpr uint QUANTIZE = 1 << 1;
  constexpr bool contains(uint flag) const thread { return (raw_value & flag) != 0; }
  constexpr bool contains(uint flag) const constant { return (raw_value & flag) != 0; }
  constexpr uint bits() const thread { return raw_value; }
  constexpr uint bits() const constant { return raw_value; }
};

enum class GemmAPrologueKind : uint32_t {
  FullPrecision = 0,
  Int8Symmetric = 1,
};

enum class ActivationScaleStatistic : uint32_t {
  AbsMax = 0,
  Rms = 1,
};
} // namespace uzu::activation_prepare

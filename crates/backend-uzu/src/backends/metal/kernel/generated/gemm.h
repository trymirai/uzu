// Auto-generated from gpu_types/gemm - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::gemm {
enum class GemmBPrologueKind : uint32_t {
  FullPrecision = 0,
  ScaleBiasDequant = 1,
  ScaleZeroPointDequant = 2,
};

struct GemmDTransform {
  uint raw_value;
  constexpr GemmDTransform() thread : raw_value(0) {}
  constexpr GemmDTransform(uint __dsl_v) thread : raw_value(__dsl_v) {}
  static constant constexpr uint SCALE = 1 << 0;
  static constant constexpr uint ACCUMULATE = 1 << 1;
  static constant constexpr uint BIAS = 1 << 2;
  static constant constexpr uint RHT = 1 << 3;
  constexpr bool contains(uint flag) const thread {
    return (raw_value & flag) != 0;
  }
  constexpr bool contains(uint flag) const constant {
    return (raw_value & flag) != 0;
  }
  constexpr uint bits() const thread { return raw_value; }
  constexpr uint bits() const constant { return raw_value; }
};

enum class GemmTiling : uint32_t {
  T8x32x32_1x1 = 0,
  T64x32x32_2x2 = 1,
  T64x64x16_2x2 = 2,
  T64x64x32_2x2 = 3,
  T64x64x64_2x2 = 4,
  T32x32x32_2x2 = 5,
  T32x64x32_2x2 = 6,
  T64x32x32_4x1 = 7,
  T128x128x32_4x4 = 8,
};

struct GemmAlignment {
  uint raw_value;
  constexpr GemmAlignment() thread : raw_value(0) {}
  constexpr GemmAlignment(uint __dsl_v) thread : raw_value(__dsl_v) {}
  static constant constexpr uint M = 1 << 0;
  static constant constexpr uint N = 1 << 1;
  static constant constexpr uint K = 1 << 2;
  constexpr bool contains(uint flag) const thread {
    return (raw_value & flag) != 0;
  }
  constexpr bool contains(uint flag) const constant {
    return (raw_value & flag) != 0;
  }
  constexpr uint bits() const thread { return raw_value; }
  constexpr uint bits() const constant { return raw_value; }
};
} // namespace uzu::gemm

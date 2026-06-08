// Auto-generated from gpu_types/gemm - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::gemm {
enum class GemmBPrologueKind : uint32_t {
  FullPrecision = 0,
  ScaleBiasDequant = 1,
  ScaleZeroPointDequant = 2,
  ScaleSymmetricDequant = 3,
  LloydMaxDequant = 4,
};

struct GemmDTransform {
  uint raw_value;
  constexpr GemmDTransform() thread : raw_value(0) {}
  constexpr GemmDTransform(uint __dsl_v) thread : raw_value(__dsl_v) {}
  static constant constexpr uint SCALE = 1 << 0;
  static constant constexpr uint ACCUMULATE = 1 << 1;
  static constant constexpr uint BIAS = 1 << 2;
  static constant constexpr uint RHT = 1 << 3;
  constexpr bool contains(uint flag) const thread { return (raw_value & flag) != 0; }
  constexpr bool contains(uint flag) const constant { return (raw_value & flag) != 0; }
  constexpr uint bits() const thread { return raw_value; }
  constexpr uint bits() const constant { return raw_value; }
};

enum class GemmTiling : uint32_t {
  Tile8x32x32_Simdgroups1x1 = 0,
  Tile64x32x32_Simdgroups2x2 = 1,
  Tile64x64x16_Simdgroups2x2 = 2,
  Tile64x64x32_Simdgroups2x2 = 3,
  Tile32x32x32_Simdgroups2x2 = 4,
  Tile32x64x256_Simdgroups2x2 = 5,
  Tile64x32x256_Simdgroups4x1 = 6,
  Tile64x64x256_Simdgroups2x2 = 7,
  Tile128x128x256_Simdgroups4x4 = 8,
};

struct GemmAlignment {
  uint raw_value;
  constexpr GemmAlignment() thread : raw_value(0) {}
  constexpr GemmAlignment(uint __dsl_v) thread : raw_value(__dsl_v) {}
  static constant constexpr uint M = 1 << 0;
  static constant constexpr uint N = 1 << 1;
  static constant constexpr uint K = 1 << 2;
  constexpr bool contains(uint flag) const thread { return (raw_value & flag) != 0; }
  constexpr bool contains(uint flag) const constant { return (raw_value & flag) != 0; }
  constexpr uint bits() const thread { return raw_value; }
  constexpr uint bits() const constant { return raw_value; }
};
} // namespace uzu::gemm

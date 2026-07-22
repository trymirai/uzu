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
};

struct GemmDTransform {
  uint32_t raw_value;
  constexpr GemmDTransform() thread : raw_value(0) {}
  constexpr GemmDTransform(uint32_t __dsl_v) thread : raw_value(__dsl_v) {}
  static constant constexpr uint32_t SCALE = 1 << 0;
  static constant constexpr uint32_t ACCUMULATE = 1 << 1;
  static constant constexpr uint32_t BIAS = 1 << 2;
  static constant constexpr uint32_t RHT = 1 << 3;
  static constant constexpr uint32_t SOFT_CAP = 1 << 4;
  constexpr bool contains(uint32_t flag) const thread { return (raw_value & flag) != 0; }
  constexpr bool contains(uint32_t flag) const constant { return (raw_value & flag) != 0; }
  constexpr uint32_t bits() const thread { return raw_value; }
  constexpr uint32_t bits() const constant { return raw_value; }
};

enum class GemmTiling : uint32_t {
  Tile8x32x32_Simdgroups1x1 = 0,
  Tile64x32x32_Simdgroups2x2 = 1,
  Tile64x64x16_Simdgroups2x2 = 2,
  Tile64x64x32_Simdgroups2x2 = 3,
  Tile32x32x32_Simdgroups2x2 = 4,
  Tile16x32x256_Simdgroups1x1 = 5,
  Tile16x128x256_Simdgroups1x4 = 6,
  Tile32x64x256_Simdgroups2x2 = 7,
  Tile64x32x256_Simdgroups4x1 = 8,
  Tile64x64x256_Simdgroups2x2 = 9,
  Tile128x128x256_Simdgroups4x4 = 10,
};

static constant constexpr uint32_t MXU_SIMDGROUP_BLOCK_K = 32;

struct GemmAlignment {
  uint32_t raw_value;
  constexpr GemmAlignment() thread : raw_value(0) {}
  constexpr GemmAlignment(uint32_t __dsl_v) thread : raw_value(__dsl_v) {}
  static constant constexpr uint32_t M = 1 << 0;
  static constant constexpr uint32_t N = 1 << 1;
  static constant constexpr uint32_t K = 1 << 2;
  constexpr bool contains(uint32_t flag) const thread { return (raw_value & flag) != 0; }
  constexpr bool contains(uint32_t flag) const constant { return (raw_value & flag) != 0; }
  constexpr uint32_t bits() const thread { return raw_value; }
  constexpr uint32_t bits() const constant { return raw_value; }
};
} // namespace uzu::gemm

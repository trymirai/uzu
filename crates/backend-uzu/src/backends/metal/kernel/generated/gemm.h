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
  uint raw_value;
  constexpr GemmDTransform() thread : raw_value(0) {}
  constexpr GemmDTransform(uint __dsl_v) thread : raw_value(__dsl_v) {}
  static constant constexpr uint SCALE = 1 << 0;
  static constant constexpr uint ACCUMULATE = 1 << 1;
  static constant constexpr uint BIAS = 1 << 2;
  static constant constexpr uint RHT = 1 << 3;
  static constant constexpr uint SOFT_CAP = 1 << 4;
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
  Tile16x32x256_Simdgroups1x1 = 5,
  Tile16x128x256_Simdgroups1x4 = 6,
  Tile32x64x256_Simdgroups2x2 = 7,
  Tile64x32x256_Simdgroups4x1 = 8,
  Tile64x64x256_Simdgroups2x2 = 9,
  Tile128x128x256_Simdgroups4x4 = 10,
};

constexpr uint gemm_tiling_block_m(GemmTiling t) {
  return
      t == GemmTiling::Tile8x32x32_Simdgroups1x1 ? 8
    :
      t == GemmTiling::Tile64x32x32_Simdgroups2x2 ? 64
    :
      t == GemmTiling::Tile64x64x16_Simdgroups2x2 ? 64
    :
      t == GemmTiling::Tile64x64x32_Simdgroups2x2 ? 64
    :
      t == GemmTiling::Tile32x32x32_Simdgroups2x2 ? 32
    :
      t == GemmTiling::Tile16x32x256_Simdgroups1x1 ? 16
    :
      t == GemmTiling::Tile16x128x256_Simdgroups1x4 ? 16
    :
      t == GemmTiling::Tile32x64x256_Simdgroups2x2 ? 32
    :
      t == GemmTiling::Tile64x32x256_Simdgroups4x1 ? 64
    :
      t == GemmTiling::Tile64x64x256_Simdgroups2x2 ? 64
    :
      t == GemmTiling::Tile128x128x256_Simdgroups4x4 ? 128
    : 0;
}

constexpr uint gemm_tiling_block_n(GemmTiling t) {
  return
      t == GemmTiling::Tile8x32x32_Simdgroups1x1 ? 32
    :
      t == GemmTiling::Tile64x32x32_Simdgroups2x2 ? 32
    :
      t == GemmTiling::Tile64x64x16_Simdgroups2x2 ? 64
    :
      t == GemmTiling::Tile64x64x32_Simdgroups2x2 ? 64
    :
      t == GemmTiling::Tile32x32x32_Simdgroups2x2 ? 32
    :
      t == GemmTiling::Tile16x32x256_Simdgroups1x1 ? 32
    :
      t == GemmTiling::Tile16x128x256_Simdgroups1x4 ? 128
    :
      t == GemmTiling::Tile32x64x256_Simdgroups2x2 ? 64
    :
      t == GemmTiling::Tile64x32x256_Simdgroups4x1 ? 32
    :
      t == GemmTiling::Tile64x64x256_Simdgroups2x2 ? 64
    :
      t == GemmTiling::Tile128x128x256_Simdgroups4x4 ? 128
    : 0;
}

constexpr uint gemm_tiling_block_k(GemmTiling t) {
  return
      t == GemmTiling::Tile8x32x32_Simdgroups1x1 ? 32
    :
      t == GemmTiling::Tile64x32x32_Simdgroups2x2 ? 32
    :
      t == GemmTiling::Tile64x64x16_Simdgroups2x2 ? 16
    :
      t == GemmTiling::Tile64x64x32_Simdgroups2x2 ? 32
    :
      t == GemmTiling::Tile32x32x32_Simdgroups2x2 ? 32
    :
      t == GemmTiling::Tile16x32x256_Simdgroups1x1 ? 256
    :
      t == GemmTiling::Tile16x128x256_Simdgroups1x4 ? 256
    :
      t == GemmTiling::Tile32x64x256_Simdgroups2x2 ? 256
    :
      t == GemmTiling::Tile64x32x256_Simdgroups4x1 ? 256
    :
      t == GemmTiling::Tile64x64x256_Simdgroups2x2 ? 256
    :
      t == GemmTiling::Tile128x128x256_Simdgroups4x4 ? 256
    : 0;
}

constexpr uint gemm_tiling_simdgroups_per_row(GemmTiling t) {
  return
      t == GemmTiling::Tile8x32x32_Simdgroups1x1 ? 1
    :
      t == GemmTiling::Tile64x32x32_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile64x64x16_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile64x64x32_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile32x32x32_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile16x32x256_Simdgroups1x1 ? 1
    :
      t == GemmTiling::Tile16x128x256_Simdgroups1x4 ? 1
    :
      t == GemmTiling::Tile32x64x256_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile64x32x256_Simdgroups4x1 ? 4
    :
      t == GemmTiling::Tile64x64x256_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile128x128x256_Simdgroups4x4 ? 4
    : 0;
}

constexpr uint gemm_tiling_simdgroups_per_column(GemmTiling t) {
  return
      t == GemmTiling::Tile8x32x32_Simdgroups1x1 ? 1
    :
      t == GemmTiling::Tile64x32x32_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile64x64x16_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile64x64x32_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile32x32x32_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile16x32x256_Simdgroups1x1 ? 1
    :
      t == GemmTiling::Tile16x128x256_Simdgroups1x4 ? 4
    :
      t == GemmTiling::Tile32x64x256_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile64x32x256_Simdgroups4x1 ? 1
    :
      t == GemmTiling::Tile64x64x256_Simdgroups2x2 ? 2
    :
      t == GemmTiling::Tile128x128x256_Simdgroups4x4 ? 4
    : 0;
}

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

enum class QuantBits : uint32_t {
  B4 = 4,
  B8 = 8,
};

enum class QuantGroupSize : uint32_t {
  G16 = 16,
  G32 = 32,
  G64 = 64,
  G128 = 128,
};

enum class QuantPrologue : uint32_t {
  ScaleBiasDequant = 1,
  ScaleZeroPointDequant = 2,
  ScaleSymmetricDequant = 3,
};
} // namespace uzu::gemm

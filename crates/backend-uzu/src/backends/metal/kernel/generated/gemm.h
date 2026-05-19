// Auto-generated from gpu_types/gemm - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::gemm {
enum class GemmInputPrologueKind : uint32_t {
  FullPrecision = 0,
  ExternalRht = 1,
};

enum class GemmOutputTransformKind : uint32_t {
  Store = 0,
  Scale = 1,
  Accumulate = 2,
  Bias = 3,
  Rht = 4,
  ScaleAccumulate = 5,
  ScaleAccumulateBias = 6,
  ScaleAccumulateBiasRht = 7,
};

enum class GemmWeightPrologueKind : uint32_t {
  FullPrecision = 0,
  ScaleBiasDequant = 1,
  ScaleZeroPointDequant = 2,
};

enum class GemmTiling : uint32_t {
  T64x32x32_2x2 = 0,
  T64x64x16_2x2 = 1,
  T64x64x32_2x2 = 2,
  T32x32x32_2x2 = 3,
  T32x64x32_2x2 = 4,
  T64x32x32_4x1 = 5,
  T128x128x32_4x4 = 6,
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
};
} // namespace uzu::gemm

// Auto-generated from gpu_types/gemm - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::gemm {
enum class GemmComputeKind : uint32_t {
  SimdgroupMma = 0,
  MxuMma = 1,
};

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

typedef struct {
  uint32_t threadgroup_m;
  uint32_t threadgroup_n;
  uint32_t threadgroup_k;
  uint32_t simdgroups_m;
  uint32_t simdgroups_n;
} GemmTilingConfig;

struct GemmAlignment {
  uint raw_value;
  constexpr GemmAlignment() thread : raw_value(0) {}
  constexpr GemmAlignment(uint __dsl_v) thread : raw_value(__dsl_v) {}
  static constant constexpr uint M = 1 << 0;
  static constant constexpr uint N = 1 << 1;
  static constant constexpr uint K = 1 << 2;
  constexpr bool contains(uint flag) const thread { return (raw_value & flag) != 0; }
};
} // namespace uzu::gemm

// Auto-generated from gpu_types/unified_gemm - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::unified_gemm {
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

typedef struct {
  uint32_t threadgroup_m;
  uint32_t threadgroup_n;
  uint32_t threadgroup_k;
  uint32_t simdgroup_m;
  uint32_t simdgroup_n;
  uint32_t simdgroup_k;
  uint32_t fragment_m;
  uint32_t fragment_n;
  uint32_t fragment_k;
  uint32_t simdgroups_m;
  uint32_t simdgroups_n;
} GemmTilingConfig;

enum class GemmWeightPrologueKind : uint32_t {
  FullPrecision = 0,
  ScaleBiasDequant = 1,
  ScaleZeroPointDequant = 2,
};
} // namespace uzu::unified_gemm

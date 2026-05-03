// Auto-generated from gpu_types/unified_gemm - do not edit manually
#pragma once

#include <metal_stdlib>
using namespace metal;

namespace uzu::unified_gemm {
struct alignas(4) GemmAlignment {
  bool m_aligned;
  bool n_aligned;
  bool k_aligned;
  inline GemmAlignment() = default;
  inline GemmAlignment(uint __dsl_v)
    : m_aligned(uint8_t(__dsl_v))
    , n_aligned(uint8_t(__dsl_v >> 8))
    , k_aligned(uint8_t(__dsl_v >> 16))
  {}
};

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
  MlxDequant = 1,
  AwqDequant = 2,
};
} // namespace uzu::unified_gemm

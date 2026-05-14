#pragma once

#include "../../common/simdgroup_mma_core.h"

namespace uzu {
namespace unified_gemm {

template <
    typename T,
    uint THREADGROUP_M,
    uint THREADGROUP_N,
    uint THREADGROUP_K,
    uint SIMDGROUPS_M,
    uint SIMDGROUPS_N,
    bool MN_ALIGNED,
    bool K_ALIGNED>
struct GemmComputeSimdgroupMma {
  static METAL_FUNC void run(
      const device T* activations,
      const device T* weights,
      device T* result,
      const constant uzu::matmul::GemmParams* params,
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      uint simd_lane_id,
      uint simd_group_id,
      uint2 threadgroup_position,
      uint3 thread_position
  ) {
    SimdgroupMmaCore<
        T,
        THREADGROUP_M,
        THREADGROUP_N,
        THREADGROUP_K,
        SIMDGROUPS_N,
        SIMDGROUPS_M,
        MN_ALIGNED,
        K_ALIGNED>::
        run(activations,
            weights,
            result,
            params,
            a_shared,
            b_shared,
            simd_lane_id,
            simd_group_id,
            threadgroup_position,
            thread_position);
  }
};

} // namespace unified_gemm
} // namespace uzu

#pragma once

#include "../../common/simdgroup_mma_core.h"

namespace uzu {
namespace gemm {

template <
    typename T,
    uint THREADGROUP_M,
    uint THREADGROUP_N,
    uint THREADGROUP_K,
    uint SIMDGROUPS_M,
    uint SIMDGROUPS_N>
struct GemmComputeSimdgroupMma {
  static METAL_FUNC void run(
      const device T* activations,
      const device T* weights,
      device T* result,
      const constant uzu::matmul::GemmParams* params,
      const bool align_m,
      const bool align_n,
      const bool align_k,
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      uint2 threadgroup_position,
      const thread ThreadContext& thread_context
  ) {
    SimdgroupMmaCore<
        T,
        THREADGROUP_M,
        THREADGROUP_N,
        THREADGROUP_K,
        SIMDGROUPS_N,
        SIMDGROUPS_M>::
        run(activations,
            weights,
            result,
            params,
            align_m,
            align_n,
            align_k,
            a_shared,
            b_shared,
            threadgroup_position,
            thread_context);
  }
};

} // namespace gemm
} // namespace uzu

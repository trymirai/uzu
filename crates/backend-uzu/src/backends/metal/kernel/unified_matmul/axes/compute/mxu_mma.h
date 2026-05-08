#pragma once

#include "../../common/mxu_mma_core.h"

namespace uzu {
namespace unified_gemm {

template <
    typename T,
    uint THREADGROUP_M,
    uint THREADGROUP_N,
    uint SIMDGROUPS_M,
    uint SIMDGROUPS_N,
    bool VALID = (THREADGROUP_M % SIMDGROUPS_N == 0 &&
                  THREADGROUP_N % SIMDGROUPS_M == 0 &&
                  (THREADGROUP_M / SIMDGROUPS_N) % 16 == 0 &&
                  (THREADGROUP_N / SIMDGROUPS_M) % 16 == 0)>
struct GemmComputeMxuMma {
  static METAL_FUNC void run(
      const device T* activations,
      const device T* weights,
      device T* result,
      const constant uzu::matmul::GemmParams* params,
      bool align_m,
      bool align_n,
      bool align_k,
      uint simd_group_id,
      uint2 threadgroup_position,
      const thread ThreadContext& thread_context
  ) {
    MxuMmaCore<
        T,
        THREADGROUP_M,
        THREADGROUP_N,
        SIMDGROUPS_N,
        SIMDGROUPS_M>::run(
        activations,
        weights,
        result,
        params,
        align_m,
        align_n,
        align_k,
        simd_group_id,
        threadgroup_position,
        thread_context);
  }
};

template <typename T, uint THREADGROUP_M, uint THREADGROUP_N, uint SIMDGROUPS_M, uint SIMDGROUPS_N>
struct GemmComputeMxuMma<T, THREADGROUP_M, THREADGROUP_N, SIMDGROUPS_M, SIMDGROUPS_N, false> {
  static METAL_FUNC void run(
      const device T* activations,
      const device T* weights,
      device T* result,
      const constant uzu::matmul::GemmParams* params,
      bool align_m,
      bool align_n,
      bool align_k,
      uint simd_group_id,
      uint2 threadgroup_position,
      const thread ThreadContext& thread_context
  ) {
    (void)activations;
    (void)weights;
    (void)result;
    (void)params;
    (void)align_m;
    (void)align_n;
    (void)align_k;
    (void)simd_group_id;
    (void)threadgroup_position;
    (void)thread_context;
  }
};

} // namespace unified_gemm
} // namespace uzu

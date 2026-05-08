#pragma once

#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../axes/compute/mxu_mma.h"
#include "../axes/compute/simdgroup_mma.h"
#include "../generated/unified_gemm.h"

namespace uzu {
namespace unified_gemm {

template <
    typename T,
    uint THREADGROUP_M,
    uint THREADGROUP_N,
    uint THREADGROUP_K,
    uint SIMDGROUPS_M,
    uint SIMDGROUPS_N>
struct GemmPipeline {
  static METAL_FUNC void run(
      const device T* activations,
      const device uint8_t* weights_packed,
      device T* result,
      const constant uzu::matmul::GemmParams* params,
      const constant float& ab_scale,
      GemmInputPrologueKind input_prologue,
      GemmWeightPrologueKind weight_prologue,
      GemmComputeKind compute,
      GemmOutputTransformKind output_transform,
      GemmAlignment alignment,
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      uint simd_lane_id,
      uint simd_group_id,
      uint2 threadgroup_position,
      uint3 thread_position,
      const thread ThreadContext& thread_context
  ) {
    if (weight_prologue != GemmWeightPrologueKind::FullPrecision ||
        input_prologue != GemmInputPrologueKind::FullPrecision ||
        output_transform != GemmOutputTransformKind::Store) {
      return;
    }
    const device T* weights = reinterpret_cast<const device T*>(weights_packed);
    if (compute == GemmComputeKind::SimdgroupMma) {
      const bool mn_aligned = alignment.m_aligned && alignment.n_aligned;
      const bool k_aligned = alignment.k_aligned;
      if (mn_aligned && k_aligned) {
        GemmComputeSimdgroupMma<T, THREADGROUP_M, THREADGROUP_N, THREADGROUP_K, SIMDGROUPS_M, SIMDGROUPS_N, true, true>::run(
            activations, weights, result, params, ab_scale,
            a_shared, b_shared, simd_lane_id, simd_group_id,
            threadgroup_position, thread_position);
      } else if (mn_aligned && !k_aligned) {
        GemmComputeSimdgroupMma<T, THREADGROUP_M, THREADGROUP_N, THREADGROUP_K, SIMDGROUPS_M, SIMDGROUPS_N, true, false>::run(
            activations, weights, result, params, ab_scale,
            a_shared, b_shared, simd_lane_id, simd_group_id,
            threadgroup_position, thread_position);
      } else if (!mn_aligned && k_aligned) {
        GemmComputeSimdgroupMma<T, THREADGROUP_M, THREADGROUP_N, THREADGROUP_K, SIMDGROUPS_M, SIMDGROUPS_N, false, true>::run(
            activations, weights, result, params, ab_scale,
            a_shared, b_shared, simd_lane_id, simd_group_id,
            threadgroup_position, thread_position);
      } else {
        GemmComputeSimdgroupMma<T, THREADGROUP_M, THREADGROUP_N, THREADGROUP_K, SIMDGROUPS_M, SIMDGROUPS_N, false, false>::run(
            activations, weights, result, params, ab_scale,
            a_shared, b_shared, simd_lane_id, simd_group_id,
            threadgroup_position, thread_position);
      }
    } else if (compute == GemmComputeKind::MxuMma) {
      GemmComputeMxuMma<T, THREADGROUP_M, THREADGROUP_N, SIMDGROUPS_M, SIMDGROUPS_N>::run(
          activations, weights, result, params, ab_scale,
          alignment.m_aligned, alignment.n_aligned, alignment.k_aligned,
          simd_group_id, threadgroup_position, thread_context);
    }
  }
};

} // namespace unified_gemm
} // namespace uzu

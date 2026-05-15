#pragma once

#include "../../../common/dsl.h"
#include "../../../common/thread_context.h"
#include "../axes/compute/mxu_mma.h"
#include "../axes/compute/simdgroup_mma.h"
#include "../generated/gemm.h"

namespace uzu {
namespace gemm {

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
      GemmInputPrologueKind input_prologue,
      GemmWeightPrologueKind weight_prologue,
      GemmComputeKind compute,
      GemmOutputTransformKind output_transform,
      GemmAlignment alignment,
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      uint2 threadgroup_position,
      const thread ThreadContext& thread_context
  ) {
    if (weight_prologue != GemmWeightPrologueKind::FullPrecision ||
        input_prologue != GemmInputPrologueKind::FullPrecision ||
        output_transform != GemmOutputTransformKind::Store) {
      return;
    }
    const bool m_aligned = alignment.contains(GemmAlignment::M);
    const bool n_aligned = alignment.contains(GemmAlignment::N);
    const bool k_aligned = alignment.contains(GemmAlignment::K);
    const device T* weights = reinterpret_cast<const device T*>(weights_packed);
    if (compute == GemmComputeKind::SimdgroupMma) {
      GemmComputeSimdgroupMma<
          T,
          THREADGROUP_M,
          THREADGROUP_N,
          THREADGROUP_K,
          SIMDGROUPS_M,
          SIMDGROUPS_N>::
          run(activations,
              weights,
              result,
              params,
              m_aligned,
              n_aligned,
              k_aligned,
              a_shared,
              b_shared,
              threadgroup_position,
              thread_context);
    } else if (compute == GemmComputeKind::MxuMma) {
      GemmComputeMxuMma<
          T,
          THREADGROUP_M,
          THREADGROUP_N,
          256,
          SIMDGROUPS_M,
          SIMDGROUPS_N>::
          run(activations,
              weights,
              result,
              params,
              m_aligned,
              n_aligned,
              k_aligned,
              thread_context.simd_lane_id,
              threadgroup_position,
              thread_context);
    }
  }
};

} // namespace gemm
} // namespace uzu

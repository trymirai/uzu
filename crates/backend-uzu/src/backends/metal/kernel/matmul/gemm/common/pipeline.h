#pragma once

#include "../../../common/dsl.h"
#include "../../../common/thread_context.h"
#include "../generated/gemm.h"
#include "mxu_mma_core.h"
#include "simdgroup_mma_core.h"

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
    // Weight/input prologues other than FullPrecision are scaffolding for the
    // upcoming quant unification — not yet implemented in the cores.
    if (weight_prologue != GemmWeightPrologueKind::FullPrecision ||
        input_prologue != GemmInputPrologueKind::FullPrecision) {
      return;
    }
    const bool m_aligned = alignment.contains(GemmAlignment::M);
    const bool n_aligned = alignment.contains(GemmAlignment::N);
    const bool k_aligned = alignment.contains(GemmAlignment::K);
    const device T* weights = reinterpret_cast<const device T*>(weights_packed);
    if (compute == GemmComputeKind::SimdgroupMma) {
      SimdgroupMmaCore<
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
              output_transform,
              a_shared,
              b_shared,
              threadgroup_position,
              thread_context);
    } else if (compute == GemmComputeKind::MxuMma) {
      MxuMmaCore<
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
              output_transform,
              thread_context.simdgroup_index,
              threadgroup_position,
              thread_context);
    }
  }
};

} // namespace gemm
} // namespace uzu

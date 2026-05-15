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
    uint SIMDGROUPS_N,
    bool TRANSPOSE_WEIGHTS>
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
    switch (compute) {
      case GemmComputeKind::SimdgroupMma:
        SimdgroupMmaCore<
            T,
            THREADGROUP_M,
            THREADGROUP_N,
            THREADGROUP_K,
            SIMDGROUPS_M,
            SIMDGROUPS_N,
            TRANSPOSE_WEIGHTS>::
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
                thread_context);
        break;
      case GemmComputeKind::MxuMma:
        MxuMmaCore<
            T,
            THREADGROUP_M,
            THREADGROUP_N,
            256,
            SIMDGROUPS_M,
            SIMDGROUPS_N,
            TRANSPOSE_WEIGHTS>::
            run(activations,
                weights,
                result,
                params,
                m_aligned,
                n_aligned,
                k_aligned,
                output_transform,
                thread_context);
        break;
    }
  }
};

} // namespace gemm
} // namespace uzu

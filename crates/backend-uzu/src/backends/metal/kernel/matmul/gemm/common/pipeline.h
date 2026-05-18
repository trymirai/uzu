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
    uint THREADGROUP_BLOCK_M,
    uint THREADGROUP_BLOCK_N,
    uint THREADGROUP_BLOCK_K,
    uint SIMDGROUPS_M,
    uint SIMDGROUPS_N,
    bool TRANSPOSE_B>
struct GemmPipeline {
  static METAL_FUNC void run(
      const device T* a,
      const device uint8_t* b_packed,
      device T* d,
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
    // Scaffolding: only FullPrecision prologues are implemented today.
    if (weight_prologue != GemmWeightPrologueKind::FullPrecision ||
        input_prologue != GemmInputPrologueKind::FullPrecision) {
      return;
    }
    const bool m_aligned = alignment.contains(GemmAlignment::M);
    const bool n_aligned = alignment.contains(GemmAlignment::N);
    const bool k_aligned = alignment.contains(GemmAlignment::K);
    const device T* b = reinterpret_cast<const device T*>(b_packed);
    switch (compute) {
    case GemmComputeKind::SimdgroupMma:
      SimdgroupMmaCore<
          T,
          THREADGROUP_BLOCK_M,
          THREADGROUP_BLOCK_N,
          THREADGROUP_BLOCK_K,
          SIMDGROUPS_M,
          SIMDGROUPS_N,
          TRANSPOSE_B>::
          run(a,
              b,
              d,
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
          THREADGROUP_BLOCK_M,
          THREADGROUP_BLOCK_N,
          SIMDGROUPS_M,
          SIMDGROUPS_N,
          TRANSPOSE_B>::
          run(a,
              b,
              d,
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

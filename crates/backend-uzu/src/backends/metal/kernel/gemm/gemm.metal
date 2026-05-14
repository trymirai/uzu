#include "../common/dsl.h"
#include "../common/defines.h"
#include "../common/thread_context.h"
#include "../generated/gemm.h"

#include "common/pipeline.h"

using namespace metal;
using namespace uzu::gemm;

#define GEMM_MAX_THREADGROUP_A 2560
#define GEMM_MAX_THREADGROUP_B 1536

template <
    typename T,
    uint THREADGROUP_M,
    uint THREADGROUP_N,
    uint THREADGROUP_K,
    uint SIMDGROUPS_M,
    uint SIMDGROUPS_N>
VARIANTS(T, float, half, bfloat)
VARIANTS(THREADGROUP_M, 32, 64, 128)
VARIANTS(THREADGROUP_N, 32, 64, 128)
VARIANTS(THREADGROUP_K, 16, 32)
VARIANTS(SIMDGROUPS_M, 1, 2, 4)
VARIANTS(SIMDGROUPS_N, 1, 2, 4)
CONSTRAINT(max(THREADGROUP_M, THREADGROUP_N) <= 32 * SIMDGROUPS_M * SIMDGROUPS_N)
KERNEL(Gemm)(
    const device T* activations,
    const device uint8_t* weights,
    device T* result,
    const device T* scales
        OPTIONAL(weight_prologue != GemmWeightPrologueKind::FullPrecision),
    const device T* biases
        OPTIONAL(weight_prologue == GemmWeightPrologueKind::ScaleBiasDequant),
    const device uint8_t* zero_points
        OPTIONAL(weight_prologue == GemmWeightPrologueKind::ScaleZeroPointDequant),
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const GemmInputPrologueKind input_prologue SPECIALIZE,
    const GemmWeightPrologueKind weight_prologue SPECIALIZE,
    const GemmComputeKind compute SPECIALIZE,
    const GemmOutputTransformKind output_transform SPECIALIZE,
    const uint alignment SPECIALIZE,
    const uint bits_per_weight SPECIALIZE,
    const uint group_size SPECIALIZE,
    threadgroup T a_shared[GEMM_MAX_THREADGROUP_A],
    threadgroup T b_shared[GEMM_MAX_THREADGROUP_B],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(SIMDGROUPS_N),
    const uint thread_z THREADS(SIMDGROUPS_M),
    const ThreadContext thread_context
) {
  (void)scales;
  (void)biases;
  (void)zero_points;
  (void)bits_per_weight;
  (void)group_size;
  GemmPipeline<
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
          input_prologue,
          weight_prologue,
          compute,
          output_transform,
          alignment,
          a_shared,
          b_shared,
          thread_context.simdgroup_index,
          thread_context.threadgroup_index,
          uint2(group_x, group_y),
          uint3(thread_x, thread_y, thread_z),
          thread_context);
}

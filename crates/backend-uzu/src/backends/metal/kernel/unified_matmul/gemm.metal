#include "../common/dsl.h"
#include "../common/defines.h"
#include "../common/thread_context.h"
#include "../generated/unified_gemm.h"

#include "common/pipeline.h"

using namespace metal;
using namespace uzu::unified_gemm;

template <typename T, uint SIMDGROUPS_M, uint SIMDGROUPS_N>
VARIANTS(T, float, half, bfloat)
VARIANTS(SIMDGROUPS_M, 1, 2, 4)
VARIANTS(SIMDGROUPS_N, 1, 2, 4)
KERNEL(UnifiedGemm)(
    const device T* activations,
    const device uint8_t* weights,
    device T* result,
    const device T* scales OPTIONAL(use_mlx_quant || use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant GemmTilingConfig& tile,
    const GemmInputPrologueKind input_prologue SPECIALIZE,
    const GemmWeightPrologueKind weight_prologue SPECIALIZE,
    const GemmComputeKind compute SPECIALIZE,
    const GemmOutputTransformKind output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    const uint bits_per_weight SPECIALIZE,
    const uint group_size SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const bool use_zero_points SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(SIMDGROUPS_N),
    const uint thread_z THREADS(SIMDGROUPS_M),
    const ThreadContext thread_context
) {
  (void)activations;
  (void)weights;
  (void)result;
  (void)scales;
  (void)biases;
  (void)zero_points;
  (void)tile;
  (void)input_prologue;
  (void)weight_prologue;
  (void)compute;
  (void)output_transform;
  (void)alignment;
  (void)bits_per_weight;
  (void)group_size;
  (void)use_mlx_quant;
  (void)use_zero_points;
  (void)group_x;
  (void)group_y;
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;
  (void)thread_context;
  uzu::unified_gemm::GemmPipeline<T>::run();
}

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
    const device T* a,
    const device T* b,
    device T* d,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant GemmThreadgroupTile& threadgroup_tile,
    const constant GemmSimdgroupTile& simdgroup_tile,
    const constant GemmFragmentTile& fragment_tile,
    const GemmInputPrologueKind input_prologue SPECIALIZE,
    const GemmWeightPrologueKind weight_prologue SPECIALIZE,
    const GemmComputeKind compute SPECIALIZE,
    const GemmOutputTransformKind output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    const uint bits_per_weight SPECIALIZE,
    const bool signed_weights SPECIALIZE,
    const uint group_size SPECIALIZE,
    const QuantizedMetadataKind metadata_kind SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(SIMDGROUPS_N),
    const uint thread_z THREADS(SIMDGROUPS_M),
    const ThreadContext thread_context
) {
  (void)a;
  (void)b;
  (void)d;
  (void)threadgroup_tile;
  (void)simdgroup_tile;
  (void)fragment_tile;
  (void)input_prologue;
  (void)weight_prologue;
  (void)compute;
  (void)output_transform;
  (void)alignment;
  (void)bits_per_weight;
  (void)signed_weights;
  (void)group_size;
  (void)metadata_kind;
  (void)group_x;
  (void)group_y;
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;
  (void)thread_context;
  uzu::unified_gemm::GemmPipeline<T>::run();
}


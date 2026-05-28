#include "../../common/dsl.h"
#include "common/gemv_core.h"

using namespace metal;
using namespace uzu::quantization_method;
using namespace uzu::gemm;
using namespace uzu::gemv;
using namespace uzu::matmul;

template <typename T, GemmBPrologueKind B_PROLOGUE, uint GROUP_SIZE, uint BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(
    B_PROLOGUE,
    GemmBPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleBiasDequant,
    GemmBPrologueKind::ScaleZeroPointDequant)
VARIANTS(GROUP_SIZE, 0, 32, 64, 128)
VARIANTS(BITS, 0, 4, 8)
CONSTRAINT(
    (B_PROLOGUE == GemmBPrologueKind::FullPrecision) == (BITS == 0))
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
KERNEL(Gemv)(
    const device uint32_t* weights,
    const device T* scales
        OPTIONAL(B_PROLOGUE != GemmBPrologueKind::FullPrecision),
    const device uint8_t* zero_points
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant),
    const device T* biases
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant),
    const device T* input,
    device T* output,
    const device int32_t* hadamard_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device T* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const constant uzu::matmul::GemvParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const GemmDTransform output_transform SPECIALIZE,
    const GemvTiling gemv_tiling SPECIALIZE,
    threadgroup float threadgroup_memory[GEMV_MAX_THREADGROUP_MEMORY],
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint group_index_x GROUPS(group_count_x),
    const uint group_index_y GROUPS(group_count_y),
    const uint thread_index_x THREADS(METAL_SIMD_SIZE),
    const uint thread_index_y THREADS(8),
    const uint thread_index_z THREADS(1),
    const ThreadContext thread_context
) {
  (void)thread_index_z;
  (void)group_index_x;
  (void)group_index_y;
  (void)thread_index_x;
  (void)thread_index_y;

  GemvCore<T, B_PROLOGUE, BITS, GROUP_SIZE>::run(
      weights,
      scales,
      zero_points,
      biases,
      input,
      output,
      hadamard_factors,
      output_bias,
      params,
      output_transform,
      gemv_tiling,
      threadgroup_memory,
      shared_results,
      thread_context
  );
}

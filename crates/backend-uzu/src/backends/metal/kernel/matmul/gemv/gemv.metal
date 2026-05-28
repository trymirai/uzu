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
    const device T* a,
    const device uint8_t* b_packed,
    device T* d,
    const device T* scales
        OPTIONAL(B_PROLOGUE != GemmBPrologueKind::FullPrecision),
    const device T* biases
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant),
    const device uint8_t* zero_points
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant),
    const device T* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* rht_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const constant uzu::matmul::GemvParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const GemmDTransform output_transform SPECIALIZE,
    const GemvTiling gemv_tiling SPECIALIZE,
    threadgroup float partial_shared[GEMV_MAX_THREADGROUP_MEMORY],
    threadgroup float result_shared[METAL_SIMD_SIZE],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(8),
    const uint thread_z THREADS(1),
    const ThreadContext thread_context
) {
  (void)group_x;
  (void)group_y;
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;

  GemvCore<T, B_PROLOGUE, BITS, GROUP_SIZE>::run(
      a,
      b_packed,
      d,
      scales,
      biases,
      zero_points,
      output_bias,
      rht_factors,
      params,
      output_transform,
      gemv_tiling,
      partial_shared,
      result_shared,
      thread_context
  );
}

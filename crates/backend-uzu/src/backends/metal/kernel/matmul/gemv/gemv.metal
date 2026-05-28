#include "../../common/dsl.h"
#include "common/gemv_core.h"

using namespace metal;
using namespace uzu::quantization_method;
using namespace uzu::gemm;
using namespace uzu::gemv;
using namespace uzu::matmul;

template <typename T, uint GROUP_SIZE, uint BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 0, 32, 64, 128)
VARIANTS(BITS, 0, 4, 8)
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
KERNEL(Gemv)(
    const device uint32_t* weights,
    const device T* scales OPTIONAL(BITS != 0),
    const device uint8_t* zero_points
        OPTIONAL(BITS != 0 && quant_method == QuantizationMethod::ScaleZeroPoint),
    const device T* biases
        OPTIONAL(BITS != 0 && quant_method == QuantizationMethod::ScaleBias),
    const device T* input,
    device T* output,
    const device int32_t* hadamard_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device T* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const constant uzu::matmul::GemvParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const QuantizationMethod quant_method SPECIALIZE,
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

  // Derive prologue from BITS so the kernel's VARIANTS(...) set stays
  // (T, GROUP_SIZE, BITS); the gemv `CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))`
  // already enforces that BITS == 0 ↔ FullPrecision. The ScaleBias vs
  // ScaleZeroPoint sub-kind is still picked at runtime via `quant_method`.
  constexpr GemmBPrologueKind B_PROLOGUE = (BITS == 0)
      ? GemmBPrologueKind::FullPrecision
      : GemmBPrologueKind::ScaleBiasDequant;

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
      quant_method,
      output_transform,
      gemv_tiling,
      threadgroup_memory,
      shared_results,
      thread_context
  );
}

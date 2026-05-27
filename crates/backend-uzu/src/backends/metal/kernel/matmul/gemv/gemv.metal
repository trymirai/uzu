#include "../../common/dsl.h"
#include "common/gemv_core.h"

using namespace metal;
using namespace uzu::quantization_method;
using namespace uzu::gemm;
using namespace uzu::gemv;

// Unified GEMV: a single simdgroup-reduction matrix-vector kernel that handles
// both full-precision (BITS == 0) and group-quantized weights, selected at
// compile time. The body lives in GemvCore (gemv/common/gemv_core.h); this entry
// point is a thin dispatch, mirroring KERNEL(Gemm) over its *MmaCore structs.
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
    const uint tg_simd_rows SPECIALIZE,
    const uint tg_simd_cols SPECIALIZE,
    const uint sg_thread_rows SPECIALIZE,
    const uint sg_thread_cols SPECIALIZE,
    const uint thread_out_rows SPECIALIZE,
    const uint thread_out_cols SPECIALIZE,
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

  const GemvTile tile{
      tg_simd_rows,
      tg_simd_cols,
      sg_thread_rows,
      sg_thread_cols,
      thread_out_rows,
      thread_out_cols,
  };

  GemvCore<T, GROUP_SIZE, BITS>::run(
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
      tile,
      group_index_x,
      group_index_y,
      thread_index_x,
      thread_index_y,
      threadgroup_memory,
      shared_results,
      thread_context
  );
}

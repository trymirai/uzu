#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>

#include "../../../common/defines.h"
#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"
#include "../../../generated/quantization_method.h"
#include "../../../generated/gemm.h"
#include "../../../generated/matmul.h"
#include "../../common/qdot.h"
#include "../../common/quant_pack.h"
#include "gemv_tiling.h"

using namespace metal;
using namespace uzu::quantization_method;
using namespace uzu::gemm;
using namespace uzu::matmul;

// Upper bound for the full-precision threadgroup reduction scratch (in floats).
// Max config: tg_simd_cols=16, output_rows_per_tg=16, thread_out_rows=4
// → 16 * (16 + 4) = 320. Unused when tg_simd_cols == 1.
#define GEMV_MAX_THREADGROUP_MEMORY 320

// 0-safe wrappers: BITS == 0 is the in-band marker for the full-precision path,
// so we must not instantiate the quant pack helpers with a zero bit-width.
template <uint B>
inline constexpr uint qmv_pack_factor() {
  if constexpr (B == 0) {
    return 1u;
  } else {
    return get_pack_factor<B, 32>();
  }
}
template <uint B>
inline constexpr uint qmv_bytes_per_pack() {
  if constexpr (B == 0) {
    return 0u;
  } else {
    return get_bytes_per_pack<B, 32>();
  }
}

namespace uzu {
namespace gemv {

// Template signature mirrors gemm's `SimdgroupMmaCore`; `B_PROLOGUE` selects
// the dense vs quantized body, and within quantized the ScaleBias vs
// ScaleZeroPoint sub-kind — all at compile time (kernel VARIANTS).
template <
    typename T,
    GemmBPrologueKind B_PROLOGUE = GemmBPrologueKind::FullPrecision,
    int BITS = 0,
    int GROUP_SIZE = 0>
struct GemvCore {
  static METAL_FUNC void run(
      const device T* a,
      const device uint8_t* b_packed,
      device T* d,
      const device T* scales,
      const device T* biases,
      const device uint8_t* zero_points,
      const device T* output_bias,
      const device int32_t* rht_factors,
      const constant uzu::matmul::GemvParams* params,
      GemmDTransform output_transform,
      GemvTiling gemv_tiling,
      threadgroup float* partial_shared,
      threadgroup float* result_shared,
      const thread ThreadContext& thread_context
  ) {
    if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
      (void)scales;
      (void)biases;
      (void)zero_points;
      (void)result_shared;
      run_fp(
          b_packed,
          a,
          d,
          output_bias,
          rht_factors,
          params,
          output_transform,
          gemv_tiling,
          partial_shared,
          thread_context
      );
    } else {
      (void)gemv_tiling;
      (void)partial_shared;
      run_quantized(
          b_packed,
          scales,
          biases,
          zero_points,
          a,
          d,
          output_bias,
          rht_factors,
          params,
          output_transform,
          result_shared,
          thread_context
      );
    }
  }

  static METAL_FUNC void run_fp(
      const device uint8_t* b_packed,
      const device T* a,
      device T* d,
      const device T* output_bias,
      const device int32_t* rht_factors,
      const constant uzu::matmul::GemvParams* params,
      GemmDTransform output_transform,
      GemvTiling gemv_tiling,
      threadgroup float* partial_shared,
      const thread ThreadContext& thread_context
  );

  static METAL_FUNC void run_quantized(
      const device uint8_t* b_packed,
      const device T* scales,
      const device T* biases,
      const device uint8_t* zero_points,
      const device T* a,
      device T* d,
      const device T* output_bias,
      const device int32_t* rht_factors,
      const constant uzu::matmul::GemvParams* params,
      GemmDTransform output_transform,
      threadgroup float* result_shared,
      const thread ThreadContext& thread_context
  );
};

} // namespace gemv
} // namespace uzu

// Included after the class definition so the out-of-class member bodies see a
// complete GemvCore.
#include "gemv_core_fp.h"
#include "gemv_core_quantized.h"

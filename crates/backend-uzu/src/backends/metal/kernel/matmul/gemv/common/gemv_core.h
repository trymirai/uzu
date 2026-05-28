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

using namespace metal;
using namespace uzu::quantization_method;
using namespace uzu::gemm;

// Upper bound for the full-precision threadgroup reduction scratch (in floats).
// Max config: tg_simd_cols=16, output_rows_per_tg=16, thread_out_rows=4
// → 16 * (16 + 4) = 320. Unused when tg_simd_cols == 1.
#define GEMV_MAX_THREADGROUP_MEMORY 320

// 0-safe wrappers: BITS == 0 denotes the full-precision (quant-off) path, so we
// must not instantiate the quant pack helpers with a zero bit-width. The
// quantized run helper reads these.
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

// Per-threadgroup tile, forwarded from the kernel's function constants into the
// core (gemv's analogue of gemm's GemmTiling, but runtime-specialized).
struct GemvTile {
  uint tg_simd_rows;
  uint tg_simd_cols;
  uint sg_thread_rows;
  uint sg_thread_cols;
  uint thread_out_rows;
  uint thread_out_cols;
};

} // namespace gemv
} // namespace uzu

// Bodies of the FP and quantized run helpers. Kept in dedicated files so each
// path can be read in isolation; both are pulled in here so GemvCore::run
// resolves them at template-instantiation time.
#include "gemv_core_fp.h"
#include "gemv_core_quantized.h"

namespace uzu {
namespace gemv {

// Unified GEMV core: a single matrix-vector entry point that handles both
// full-precision (BITS == 0) and group-quantized weights, selected at compile
// time. Mirrors gemm's *MmaCore structs — KERNEL(Gemv) is a thin dispatch over
// this. The two paths' bodies live in `fp::run` (gemv_core_fp.h) and
// `quantized::run` (gemv_core_quantized.h); this struct is just the dispatcher.
//
// BITS == 0 is the in-band marker for the dense path (the CPU reference kernel
// can't take an enum const generic on stable Rust); the quantized sub-kind
// (scale+bias vs scale+zero-point) is picked at runtime from `quant_method`.
template <typename T, uint GROUP_SIZE, uint BITS>
struct GemvCore {
  static METAL_FUNC void run(
      const device uint32_t* weights,
      const device T* scales,
      const device uint8_t* zero_points,
      const device T* biases,
      const device T* input,
      device T* output,
      const device int32_t* hadamard_factors,
      const device T* output_bias,
      const constant uzu::matmul::GemvParams* params,
      QuantizationMethod quant_method,
      GemmDTransform output_transform,
      GemvTile tile,
      uint group_index_x,
      uint group_index_y,
      uint thread_index_x,
      uint thread_index_y,
      threadgroup float* threadgroup_memory,
      threadgroup float* shared_results,
      const thread ThreadContext& thread_context
  ) {
    if constexpr (BITS == 0) {
      // Quant-only inputs (scales/zero_points/biases/quant_method, the quant
      // shared_results scratch, and the per-thread indices) are unused on the
      // dense path — `OPTIONAL(...)` on the kernel side leaves them unbound.
      (void)scales;
      (void)zero_points;
      (void)biases;
      (void)quant_method;
      (void)thread_index_x;
      (void)thread_index_y;
      (void)shared_results;
      fp::run<T>(
          reinterpret_cast<const device T*>(weights),
          input,
          output,
          hadamard_factors,
          output_bias,
          params,
          output_transform,
          tile,
          group_index_x,
          group_index_y,
          threadgroup_memory,
          thread_context
      );
    } else {
      // FP-only inputs are unused on the quant path.
      (void)tile;
      (void)threadgroup_memory;
      (void)thread_context;
      quantized::run<T, GROUP_SIZE, BITS>(
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
          group_index_x,
          group_index_y,
          thread_index_x,
          thread_index_y,
          shared_results
      );
    }
  }
};

} // namespace gemv
} // namespace uzu

#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../common/fragment.h"
#include "../common/mxu_fragment_ops.h"

using namespace metal;

// Raw MXU MMA throughput probe. Operands live in registers for the whole inner
// loop: no memory traffic and no k-loop bookkeeping. Fragment shape matches the
// production per-simdgroup tile (2x2 fragments = 32x32), so the int8/bf16 ratio
// reflects the tensor-op rate the GEMM can actually reach.
//
// The int8 kernel also carries a drain-mode ladder to price the per-chunk scale
// epilogue in isolation:
//   0 - no drain: chained multiply-accumulate, the raw MMA rate;
//   1 - production-style drain per iteration: fragment_mm overwrites an integer
//       product fragment, then acc += row_scale * col_scale * float(product);
//   2 - the same drain with the magic-number int->float conversion instead of
//       the hardware convert.

UZU_CONST ushort PROBE_TILES = 2;
UZU_CONST uint PROBE_OPS_PER_SIMDGROUP_ITERATION = 32u * 32u * 32u * 2u;

// Exact for |x| < 2^22. A chunk product is bounded by 32 * 127 * 127 = 516,128,
// well inside. Integer add on the bit pattern, then a float subtract - no
// hardware int->float convert involved.
METAL_FUNC float magic_int_to_float(int x) { return as_type<float>(0x4B400000 + x) - 12582912.0f; }

template <typename OperandT, typename AccumT>
METAL_FUNC float mxu_rate_probe_body(
    const device OperandT* operand_seed,
    const uint seed_length,
    const uint inner_iterations,
    const uint drain_mode,
    const thread ThreadContext& thread_context
) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftFragment = uzu::matmul::Fragment<OperandT, PROBE_TILES, PROBE_TILES, Ops>;
  using RightFragment = uzu::matmul::Fragment<OperandT, PROBE_TILES, PROBE_TILES, Ops>;
  using AccumFragment = uzu::matmul::Fragment<AccumT, PROBE_TILES, PROBE_TILES, Ops>;
  using ProductFragment = uzu::matmul::Fragment<int, PROBE_TILES, PROBE_TILES, Ops>;
  using FloatAccumFragment = uzu::matmul::Fragment<float, PROBE_TILES, PROBE_TILES, Ops>;

  LeftFragment left;
  RightFragment right;

  // Seed operands uniquely per lane so no fragment is a compile-time constant.
  const uint lane_offset =
      (thread_context.threadgroup_position.x * 4u + thread_context.simdgroup_index) * 32u + thread_context.simd_lane_id;
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < LeftFragment::ELEMENTS_PER_FRAGMENT; ++i) {
    left.elements()[i] = operand_seed[(lane_offset + i) % seed_length];
    right.elements()[i] = operand_seed[(lane_offset + i * 7u + 3u) % seed_length];
  }

  float checksum = 0.0f;

  // The drain ladder multiplies into an integer product fragment, which only the
  // int8 operand path supports; bf16 always runs mode 0.
  constexpr bool has_drain_modes = metal::is_same_v<OperandT, int8_t>;

  if (!has_drain_modes || drain_mode == 0u) {
    AccumFragment accumulator;
    accumulator.clear();
    for (uint iteration = 0; iteration < inner_iterations; ++iteration) {
      Ops::template fragment_mma<false, false>(accumulator, left, right);
    }
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < AccumFragment::ELEMENTS_PER_FRAGMENT; ++i) {
      checksum += float(accumulator.elements()[i]);
    }
  } else if constexpr (has_drain_modes) {
    FloatAccumFragment accumulator;
    accumulator.clear();
    ProductFragment products;
    // Per-fragment-coordinate scales, rotated every iteration so the products
    // cannot be hoisted, mirroring the row/col line-cache lookups in the drain.
    float row_scales[PROBE_TILES] = {1.0f, 1.25f};
    float col_scales[PROBE_TILES] = {0.75f, 1.5f};
    for (uint iteration = 0; iteration < inner_iterations; ++iteration) {
      Ops::template fragment_mm<false, false>(products, left, right);
      METAL_PRAGMA_UNROLL
      for (ushort row = 0; row < PROBE_TILES; ++row) {
        METAL_PRAGMA_UNROLL
        for (ushort col = 0; col < PROBE_TILES; ++col) {
          thread auto& accumulated = accumulator.fragment_at(row, col);
          thread auto& product = products.fragment_at(row, col);
          const float scale = row_scales[row] * col_scales[col];
          METAL_PRAGMA_UNROLL
          for (ushort i = 0; i < Ops::ELEMENTS_PER_THREAD; ++i) {
            if (drain_mode == 2u) {
              accumulated[i] += scale * magic_int_to_float(product[i]);
            } else {
              accumulated[i] += scale * float(product[i]);
            }
          }
        }
      }
      row_scales[0] *= 1.0000001f;
      row_scales[1] *= 0.9999999f;
      col_scales[0] *= 1.0000001f;
      col_scales[1] *= 0.9999999f;
    }
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < FloatAccumFragment::ELEMENTS_PER_FRAGMENT; ++i) {
      checksum += accumulator.elements()[i];
    }
  }
  return checksum;
}

KERNEL(MxuRateProbeInt8)(
    const device int8_t* operand_seed,
    device float* checksums,
    const constant uint& seed_length,
    const constant uint& inner_iterations,
    const constant uint& threadgroup_count,
    const uint drain_mode SPECIALIZE,
    const uint threadgroup_index GROUPS(threadgroup_count),
    const uint thread_index_in_threadgroup THREADS(128),
    const ThreadContext thread_context
) {
  (void)threadgroup_index;
  const float checksum =
      mxu_rate_probe_body<int8_t, int>(operand_seed, seed_length, inner_iterations, drain_mode, thread_context);
  const uint global_thread_index = thread_context.threadgroup_position.x * 128u + thread_index_in_threadgroup;
  checksums[global_thread_index] = checksum;
}

KERNEL(MxuRateProbeBf16)(
    const device bfloat* operand_seed,
    device float* checksums,
    const constant uint& seed_length,
    const constant uint& inner_iterations,
    const constant uint& threadgroup_count,
    const uint threadgroup_index GROUPS(threadgroup_count),
    const uint thread_index_in_threadgroup THREADS(128),
    const ThreadContext thread_context
) {
  (void)threadgroup_index;
  const float checksum =
      mxu_rate_probe_body<bfloat, float>(operand_seed, seed_length, inner_iterations, 0u, thread_context);
  const uint global_thread_index = thread_context.threadgroup_position.x * 128u + thread_index_in_threadgroup;
  checksums[global_thread_index] = checksum;
}

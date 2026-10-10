#pragma once

#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"

using namespace metal;

// Preserve the existing BF16 round before the 32-wide transform, while avoiding
// a device-memory store, barrier, and reload between GEMM and RHT.
template <typename Output, typename Fragment>
static METAL_FUNC void apply_fragment_output_rht(
    thread Fragment& accumulator,
    const device int32_t* factors,
    ushort lane
) {
  const ushort col = ushort(Fragment::FragmentOpsType::get_position(lane).x);
  const float normalization = 1.0f / sqrt(32.0f);
  METAL_PRAGMA_UNROLL
  for (ushort row_slot = 0; row_slot < 2; ++row_slot) {
    float4 lo, hi;
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < 4; ++i) {
      lo[i] = float(Output(accumulator.fragment_at(0, 0)[row_slot * 4 + i]));
      hi[i] = float(Output(accumulator.fragment_at(0, 1)[row_slot * 4 + i]));
    }
    METAL_PRAGMA_UNROLL
    for (ushort stride = 1; stride < 4; stride <<= 1) {
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < 4; ++i) {
        if (!(i & stride)) {
          const float a = lo[i], b = lo[i + stride];
          const float c = hi[i], d = hi[i + stride];
          lo[i] = b + a;
          lo[i + stride] = a - b;
          hi[i] = d + c;
          hi[i + stride] = c - d;
        }
      }
    }
    const float4 lo4 = simd_shuffle_xor(lo, 1);
    const float4 hi4 = simd_shuffle_xor(hi, 1);
    lo = (col & 4) ? lo4 - lo : lo4 + lo;
    hi = (col & 4) ? hi4 - hi : hi4 + hi;
    const float4 lo8 = simd_shuffle_xor(lo, 8);
    const float4 hi8 = simd_shuffle_xor(hi, 8);
    lo = (col & 8) ? lo8 - lo : lo8 + lo;
    hi = (col & 8) ? hi8 - hi : hi8 + hi;
    const float4 sum = hi + lo, difference = lo - hi;
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < 4; ++i) {
      const Output a = Output(sum[i] * normalization * float(factors[col + i]));
      const Output b = Output(difference[i] * normalization * float(factors[col + 16 + i]));
      accumulator.fragment_at(0, 0)[row_slot * 4 + i] = float(a);
      accumulator.fragment_at(0, 1)[row_slot * 4 + i] = float(b);
    }
  }
}

template <typename T, typename BiasT>
static METAL_FUNC void apply_output_random_hadamard_transform(
    device T* output_block,
    const device int32_t* rht_factors_block,
    const device BiasT* bias_block,
    bool apply_bias,
    ushort tile_block_rows,
    ushort tile_block_cols,
    uint leading_dimension_d,
    ushort simdgroup_count,
    const thread ThreadContext& thread_context
) {
  const ushort stripes_per_row = tile_block_cols / HADAMARD_TRANSFORM_BLOCK_SIZE;
  const ushort simd_lane = thread_context.simd_lane_id;
  const uint total_work = uint(tile_block_rows) * uint(stripes_per_row);
  for (uint cell = thread_context.simdgroup_index; cell < total_work; cell += simdgroup_count) {
    const ushort row_local = ushort(cell / stripes_per_row);
    const ushort stripe = ushort(cell % stripes_per_row);
    const ushort col_local = stripe * ushort(HADAMARD_TRANSFORM_BLOCK_SIZE) + simd_lane;
    const size_t output_index = size_t(row_local) * size_t(leading_dimension_d) + size_t(col_local);
    T transformed =
        simdgroup_output_random_hadamard_transform(simd_lane, output_block[output_index], rht_factors_block[col_local]);
    if (apply_bias) {
      transformed = T(float(transformed) + float(bias_block[col_local]));
    }
    output_block[output_index] = transformed;
  }
}

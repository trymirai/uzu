#pragma once

#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"

using namespace metal;

// Sums the K-slice partials `[K_SPLIT][16][32]` of a SymRow2 tile, scales, rounds and applies the RHT; one lane per
// column.
template <ushort K_SPLIT, typename T>
static METAL_FUNC void apply_symrow2_output_random_hadamard_transform_from_k_partials(
    threadgroup const float* partials,
    device T* output_block,
    int32_t rht_factor,
    float row_scale,
    float bias,
    bool apply_bias,
    ushort tile_block_rows,
    uint leading_dimension_d,
    ushort simdgroup_count,
    const thread ThreadContext& thread_context
) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  static_assert(K_SPLIT == 2 || K_SPLIT == 4, "SymRow2 K split is 2 or 4");
  constexpr uint SLICE = 16 * HADAMARD_TRANSFORM_BLOCK_SIZE;
  const ushort simd_lane = thread_context.simd_lane_id;
  for (ushort row = ushort(thread_context.simdgroup_index); row < tile_block_rows; row += simdgroup_count) {
    threadgroup const float* p = partials + size_t(row) * HADAMARD_TRANSFORM_BLOCK_SIZE + simd_lane;
    float sum;
    if constexpr (K_SPLIT == 2) {
      sum = p[0] + p[SLICE];
    } else {
      sum = (p[0] + p[SLICE]) + (p[2 * SLICE] + p[3 * SLICE]);
    }
    sum *= row_scale;
    float result = float(simdgroup_output_random_hadamard_transform(simd_lane, T(sum), rht_factor));
    if (apply_bias) {
      result += bias;
    }
    output_block[size_t(row) * size_t(leading_dimension_d) + simd_lane] = T(result);
  }
}

template <typename T, typename BiasT, bool SINGLE_SIMDGROUP = false>
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
  for (uint cell = SINGLE_SIMDGROUP ? 0 : thread_context.simdgroup_index; cell < total_work;
       cell += SINGLE_SIMDGROUP ? 1 : simdgroup_count) {
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

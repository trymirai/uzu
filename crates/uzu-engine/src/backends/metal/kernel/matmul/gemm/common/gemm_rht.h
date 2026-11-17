#pragma once

#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"

using namespace metal;

template <typename T, typename BiasT, bool SINGLE_SIMDGROUP = false, typename Source>
static METAL_FUNC void apply_output_random_hadamard_transform(
    Source source,
    uint source_stride,
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
  constexpr ushort ROWS_PER_PASS = METAL_SIMD_SIZE / hadamard::LANES_PER_BLOCK;
  const ushort simd_lane = thread_context.simd_lane_id;
  const ushort lane_row = simd_lane / hadamard::LANES_PER_BLOCK;
  const ushort lane_column = simd_lane % hadamard::LANES_PER_BLOCK * hadamard::VECTOR_SIZE;
  for (ushort first_row = (SINGLE_SIMDGROUP ? 0 : thread_context.simdgroup_index) * ROWS_PER_PASS;
       first_row < tile_block_rows;
       first_row += (SINGLE_SIMDGROUP ? 1 : simdgroup_count) * ROWS_PER_PASS) {
    const ushort row = first_row + lane_row;
    const bool active = row < tile_block_rows;
    const auto source_row = source + size_t(row) * source_stride;
    device T* output_row = output_block + size_t(row) * leading_dimension_d;
    for (ushort column = lane_column; column < tile_block_cols; column += HADAMARD_TRANSFORM_BLOCK_SIZE) {
      const float4 values = active ? float4(load_vec4(source_row + column)) : 0.0f;
      const float4 rotated = hadamard::simdgroup_transform_vector(simd_lane, values, HADAMARD_TRANSFORM_BLOCK_SIZE);
      if (active) {
        auto result = vec<T, hadamard::VECTOR_SIZE>(rotated * float4(load_vec4(rht_factors_block + column)));
        if (apply_bias) {
          result = vec<T, hadamard::VECTOR_SIZE>(float4(result) + float4(load_vec4(bias_block + column)));
        }
        store_vec4(output_row + column, result);
      }
    }
  }
}

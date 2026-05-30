#pragma once

#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/thread_context.h"

using namespace metal;

static METAL_FUNC float simdgroup_hadamard_transform(
    ushort lane_index,
    float lane_value
) {
  for (ushort stride = 1; stride < METAL_SIMD_SIZE; stride <<= 1) {
    float partner_lane_value = simd_shuffle_xor(lane_value, stride);
    lane_value = (lane_index & stride) ? (partner_lane_value - lane_value)
                                       : (partner_lane_value + lane_value);
  }

  return lane_value / sqrt((float)METAL_SIMD_SIZE);
}

template <typename T>
static METAL_FUNC T simdgroup_input_random_hadamard_transform(
    ushort lane_index,
    T lane_value,
    int32_t lane_factor
) {
  return T(simdgroup_hadamard_transform(
      lane_index,
      float(lane_value) * float(lane_factor)
  ));
}

template <typename T>
static METAL_FUNC T simdgroup_output_random_hadamard_transform(
    ushort lane_index,
    T lane_value,
    int32_t lane_factor
) {
  return T(
      simdgroup_hadamard_transform(lane_index, float(lane_value)) *
      float(lane_factor)
  );
}

template <typename T>
static METAL_FUNC void apply_output_random_hadamard_transform(
    device T* output_block,
    const device int32_t* rht_factors_block,
    ushort tile_block_rows,
    ushort tile_block_cols,
    uint leading_dimension_d,
    ushort simdgroup_count,
    const thread ThreadContext& thread_context
) {
  const ushort stripes_per_row = tile_block_cols / METAL_SIMD_SIZE;
  const ushort simd_lane = thread_context.simd_lane_id;
  const uint total_work = uint(tile_block_rows) * uint(stripes_per_row);
  for (uint cell = thread_context.simdgroup_index; cell < total_work;
       cell += simdgroup_count) {
    const ushort row_local = ushort(cell / stripes_per_row);
    const ushort stripe = ushort(cell % stripes_per_row);
    const ushort col_local = stripe * ushort(METAL_SIMD_SIZE) + simd_lane;
    const size_t output_index =
        size_t(row_local) * size_t(leading_dimension_d) + size_t(col_local);
    output_block[output_index] = simdgroup_output_random_hadamard_transform(
        simd_lane,
        output_block[output_index],
        rht_factors_block[col_local]
    );
  }
}

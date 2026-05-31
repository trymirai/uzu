#pragma once

#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"

using namespace metal;

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

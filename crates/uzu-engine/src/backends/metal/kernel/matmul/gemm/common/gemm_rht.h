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
  const ushort stripes_per_row = tile_block_cols / HADAMARD_TRANSFORM_BLOCK_SIZE;
  const ushort simd_lane = thread_context.simd_lane_id;
  const uint total_work = uint(tile_block_rows) * uint(stripes_per_row);
  for (uint cell = SINGLE_SIMDGROUP ? 0 : thread_context.simdgroup_index; cell < total_work;
       cell += SINGLE_SIMDGROUP ? 1 : simdgroup_count) {
    const ushort row_local = ushort(cell / stripes_per_row);
    const ushort stripe = ushort(cell % stripes_per_row);
    const ushort col_local = stripe * ushort(HADAMARD_TRANSFORM_BLOCK_SIZE) + simd_lane;
    T transformed = simdgroup_output_random_hadamard_transform(
        simd_lane,
        source[size_t(row_local) * size_t(source_stride) + size_t(col_local)],
        rht_factors_block[col_local]
    );
    if (apply_bias) {
      transformed = T(float(transformed) + float(bias_block[col_local]));
    }
    output_block[size_t(row_local) * size_t(leading_dimension_d) + size_t(col_local)] = transformed;
  }
}

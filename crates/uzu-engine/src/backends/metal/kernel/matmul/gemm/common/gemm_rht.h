#pragma once

#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"

using namespace metal;

template <typename T, typename BiasT, bool SINGLE_SIMDGROUP = false, typename SourcePointer>
static METAL_FUNC void apply_output_random_hadamard_transform(
    SourcePointer source_block,
    device T* output_block,
    const device int32_t* rht_factors_block,
    const device BiasT* bias_block,
    bool apply_bias,
    ushort tile_block_rows,
    ushort tile_block_cols,
    uint source_leading_dimension,
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
    const size_t source_index = size_t(row_local) * size_t(source_leading_dimension) + size_t(col_local);
    const size_t output_index = size_t(row_local) * size_t(leading_dimension_d) + size_t(col_local);
    T transformed = simdgroup_output_random_hadamard_transform(
        simd_lane,
        static_cast<T>(source_block[source_index]),
        rht_factors_block[col_local]
    );
    if (apply_bias) {
      transformed = T(float(transformed) + float(bias_block[col_local]));
    }
    output_block[output_index] = transformed;
  }
}

// Keep the original in-place interface for the staged GEMM and GEMV callers.
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
  apply_output_random_hadamard_transform<T, BiasT, SINGLE_SIMDGROUP>(
      output_block,
      output_block,
      rht_factors_block,
      bias_block,
      apply_bias,
      tile_block_rows,
      tile_block_cols,
      leading_dimension_d,
      leading_dimension_d,
      simdgroup_count,
      thread_context
  );
}

#pragma once

#include "../../../common/integral_constant.h"
#include "../../../common/thread_context.h"
#include "../../common/defines.h"
#include "../../common/loader.h"
#include "../../common/threadgroup_tile.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "block_geometry.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename T,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int SIMDGROUPS_PER_ROW,
    int SIMDGROUPS_PER_COLUMN>
struct SimdgroupMmaCore {
  METAL_CONST ushort PADDING_A = 16 / sizeof(T);
  METAL_CONST ushort PADDING_B = 16 / sizeof(T);
  METAL_CONST ushort SHARED_STRIDE_A = BLOCK_K + PADDING_A;
  METAL_CONST ushort SHARED_STRIDE_B = BLOCK_K + PADDING_B;
  METAL_CONST ushort THREADGROUP_THREADS =
      SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;

  using ActivationsLoader = uzu::matmul::ThreadgroupLoader<
      T,
      BLOCK_M,
      BLOCK_K,
      SHARED_STRIDE_A,
      true,
      THREADGROUP_THREADS>;
  using WeightsLoader = uzu::matmul::ThreadgroupLoader<
      T,
      BLOCK_N,
      BLOCK_K,
      SHARED_STRIDE_B,
      true,
      THREADGROUP_THREADS>;
  using TileAccumulator = uzu::matmul::ThreadgroupTile<
      T,
      T,
      BLOCK_M,
      BLOCK_N,
      BLOCK_K,
      SIMDGROUPS_PER_ROW,
      SIMDGROUPS_PER_COLUMN,
      false,
      true,
      SHARED_STRIDE_A,
      SHARED_STRIDE_B,
      float,
      uzu::matmul::TransformNone<T, float>>;

  template <bool M_aligned, bool N_aligned, bool K_aligned>
  static METAL_FUNC void k_loop(
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      const int aligned_k_iterations,
      thread ActivationsLoader& loader_a,
      thread WeightsLoader& loader_b,
      thread TileAccumulator& accumulator,
      thread const ushort& tile_block_rows,
      thread const ushort& tile_block_cols,
      thread const ushort& leftover_block_depth
  ) {
    short2 tile_dimensions_a = short2(BLOCK_K, tile_block_rows);
    short2 tile_dimensions_b = short2(BLOCK_K, tile_block_cols);

    for (int k = 0; k < aligned_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (M_aligned) {
        loader_a.load_unsafe();
      } else {
        loader_a.load_safe(tile_dimensions_a);
      }
      if (N_aligned) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dimensions_b);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      accumulator.multiply_accumulate(a_shared, b_shared);

      loader_a.next();
      loader_b.next();
    }

    if (!K_aligned) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      short2 last_tile_dimensions_a =
          short2(leftover_block_depth, tile_block_rows);
      short2 last_tile_dimensions_b =
          short2(leftover_block_depth, tile_block_cols);

      loader_a.load_safe(last_tile_dimensions_a);
      loader_b.load_safe(last_tile_dimensions_b);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      accumulator.multiply_accumulate(a_shared, b_shared);
    }
  }

  static METAL_FUNC void run(
      const device T* activations,
      const device T* weights,
      device T* result,
      const constant uzu::matmul::GemmParams* params,
      const bool align_m,
      const bool align_n,
      const bool align_k,
      GemmOutputTransformKind output_transform,
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      uint2 threadgroup_position,
      const thread ThreadContext& thread_context
  ) {
    const uint simd_lane_id = thread_context.simd_lane_id;
    const uint simd_group_id = thread_context.simdgroup_index;

    const uint2 tile_id = block_id(threadgroup_position, params);
    const auto geometry =
        BlockGeometry<BLOCK_M, BLOCK_N>::compute(tile_id, params);
    if (geometry.out_of_bounds) {
      return;
    }

    threadgroup_barrier(mem_flags::mem_none);

    const size_t block_row_long = size_t(geometry.block_row_start);
    const size_t block_col_long = size_t(geometry.block_col_start);

    activations += block_row_long * params->leading_dimension_a;
    weights += block_col_long * params->leading_dimension_b;
    result += block_row_long * params->leading_dimension_d + block_col_long;

    thread ActivationsLoader loader_a(
        activations,
        params->leading_dimension_a,
        a_shared,
        simd_group_id,
        simd_lane_id
    );
    thread WeightsLoader loader_b(
        weights,
        params->leading_dimension_b,
        b_shared,
        simd_group_id,
        simd_lane_id
    );
    thread TileAccumulator accumulator(simd_group_id, simd_lane_id);

    const ushort tile_block_rows =
        min(BLOCK_M, ((int)params->M) - int(geometry.block_row_start));
    const ushort tile_block_cols =
        min(BLOCK_N, ((int)params->N) - int(geometry.block_col_start));
    const ushort leftover_block_depth =
        params->K - params->aligned_inner_iterations * BLOCK_K;

    const bool needs_epilogue = output_transform != GemmOutputTransformKind::Store;
    const float alpha =
        (output_transform == GemmOutputTransformKind::Scale ||
         output_transform == GemmOutputTransformKind::ScaleAccumulate)
            ? params->ab_scale
            : 1.0f;
    const float beta =
        (output_transform == GemmOutputTransformKind::Accumulate ||
         output_transform == GemmOutputTransformKind::ScaleAccumulate)
            ? 1.0f
            : 0.0f;
    uzu::matmul::TransformScaleAccumulate<float, float> epilogue(alpha, beta);

    dispatch_bool(align_k, [&](auto aligned_k) {
      dispatch_bool(
          align_m || (tile_block_rows == BLOCK_M),
          [&](auto aligned_m) {
            dispatch_bool(
                align_n || (tile_block_cols == BLOCK_N),
                [&](auto aligned_n) {
                  k_loop<aligned_m.value, aligned_n.value, aligned_k.value>(
                      a_shared,
                      b_shared,
                      params->aligned_inner_iterations,
                      loader_a,
                      loader_b,
                      accumulator,
                      tile_block_rows,
                      tile_block_cols,
                      leftover_block_depth
                  );
                  if constexpr (aligned_m.value && aligned_n.value) {
                    if (needs_epilogue) {
                      accumulator.apply_epilogue(
                          result, params->leading_dimension_d, 1, epilogue
                      );
                    }
                    accumulator.store_result(
                        result,
                        params->leading_dimension_d
                    );
                  } else {
                    if (needs_epilogue) {
                      accumulator.apply_epilogue_safe(
                          result,
                          params->leading_dimension_d,
                          1,
                          short2(tile_block_cols, tile_block_rows),
                          epilogue
                      );
                    }
                    accumulator.store_result_safe(
                        result,
                        params->leading_dimension_d,
                        short2(tile_block_cols, tile_block_rows)
                    );
                  }
                }
            );
          }
      );
    });
  }
};

} // namespace gemm
} // namespace uzu

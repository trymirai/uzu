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
    int THREADGROUP_BLOCK_M,
    int THREADGROUP_BLOCK_N,
    int THREADGROUP_BLOCK_K,
    int SIMDGROUPS_PER_ROW,
    int SIMDGROUPS_PER_COLUMN,
    bool TRANSPOSE_B>
struct SimdgroupMmaCore {
  METAL_CONST ushort PADDING_A = 16 / sizeof(T);
  METAL_CONST ushort PADDING_B = 16 / sizeof(T);
  METAL_CONST ushort SHARED_STRIDE_A = THREADGROUP_BLOCK_K + PADDING_A;
  METAL_CONST ushort SHARED_STRIDE_B =
      (TRANSPOSE_B ? THREADGROUP_BLOCK_K : THREADGROUP_BLOCK_N) + PADDING_B;
  METAL_CONST ushort THREADGROUP_THREADS =
      SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;

  using ALoader = uzu::matmul::ThreadgroupLoader<
      T,
      THREADGROUP_BLOCK_M,
      THREADGROUP_BLOCK_K,
      SHARED_STRIDE_A,
      true,
      THREADGROUP_THREADS>;
  using BLoader = uzu::matmul::ThreadgroupLoader<
      T,
      TRANSPOSE_B ? THREADGROUP_BLOCK_N : THREADGROUP_BLOCK_K,
      TRANSPOSE_B ? THREADGROUP_BLOCK_K : THREADGROUP_BLOCK_N,
      SHARED_STRIDE_B,
      TRANSPOSE_B,
      THREADGROUP_THREADS>;
  using TileAccumulator = uzu::matmul::ThreadgroupTile<
      T,
      T,
      THREADGROUP_BLOCK_M,
      THREADGROUP_BLOCK_N,
      THREADGROUP_BLOCK_K,
      SIMDGROUPS_PER_ROW,
      SIMDGROUPS_PER_COLUMN,
      false,
      TRANSPOSE_B,
      SHARED_STRIDE_A,
      SHARED_STRIDE_B,
      float,
      uzu::matmul::TransformNone<T, float>>;

  template <bool M_aligned, bool N_aligned, bool K_aligned>
  static METAL_FUNC void k_loop(
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      const int aligned_k_iterations,
      thread ALoader& loader_a,
      thread BLoader& loader_b,
      thread TileAccumulator& accumulator,
      thread const ushort& tile_block_rows,
      thread const ushort& tile_block_cols,
      thread const ushort& leftover_block_depth
  ) {
    short2 tile_dimensions_a = short2(THREADGROUP_BLOCK_K, tile_block_rows);
    short2 tile_dimensions_b =
        TRANSPOSE_B ? short2(THREADGROUP_BLOCK_K, tile_block_cols)
                    : short2(tile_block_cols, THREADGROUP_BLOCK_K);

    for (int k = 0; k < aligned_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if constexpr (M_aligned) {
        loader_a.load_unsafe();
      } else {
        loader_a.load_safe(tile_dimensions_a);
      }
      if constexpr (N_aligned) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dimensions_b);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      accumulator.multiply_accumulate(a_shared, b_shared);

      loader_a.next();
      loader_b.next();
    }

    if constexpr (!K_aligned) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      short2 last_tile_dimensions_a =
          short2(leftover_block_depth, tile_block_rows);
      short2 last_tile_dimensions_b =
          TRANSPOSE_B ? short2(leftover_block_depth, tile_block_cols)
                      : short2(tile_block_cols, leftover_block_depth);

      loader_a.load_safe(last_tile_dimensions_a);
      loader_b.load_safe(last_tile_dimensions_b);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      accumulator.multiply_accumulate(a_shared, b_shared);
    }
  }

  static METAL_FUNC void run(
      const device T* a,
      const device T* b,
      device T* d,
      const constant uzu::matmul::GemmParams* params,
      GemmAlignment alignment,
      GemmOutputTransformKind output_transform,
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      const thread ThreadContext& thread_context
  ) {
    const uint simd_lane_id = thread_context.simd_lane_id;
    const uint simd_group_id = thread_context.simdgroup_index;

    const uint2 tile_id =
        block_id(thread_context.threadgroup_position.xy, params);
    const auto geometry =
        ThreadgroupTileGeometry<THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N>::
            compute(tile_id, params);
    if (geometry.out_of_bounds) {
      return;
    }

    threadgroup_barrier(mem_flags::mem_none);

    const size_t block_row = size_t(geometry.block_row_start);
    const size_t block_col = size_t(geometry.block_col_start);

    a += block_row * params->leading_dimension_a;
    b += TRANSPOSE_B ? block_col * params->leading_dimension_b : block_col;
    d += block_row * params->leading_dimension_d + block_col;

    thread ALoader loader_a(
        a,
        params->leading_dimension_a,
        a_shared,
        simd_group_id,
        simd_lane_id
    );
    thread BLoader loader_b(
        b,
        params->leading_dimension_b,
        b_shared,
        simd_group_id,
        simd_lane_id
    );
    thread TileAccumulator accumulator(simd_group_id, simd_lane_id);

    const ushort tile_block_rows =
        min(THREADGROUP_BLOCK_M,
            ((int)params->M) - int(geometry.block_row_start));
    const ushort tile_block_cols =
        min(THREADGROUP_BLOCK_N,
            ((int)params->N) - int(geometry.block_col_start));
    const ushort leftover_block_depth =
        params->K - params->aligned_inner_iterations * THREADGROUP_BLOCK_K;

    const bool needs_epilogue =
        output_transform != GemmOutputTransformKind::Store;
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

    dispatch_bool(alignment.contains(GemmAlignment::K), [&](auto aligned_k) {
      dispatch_bool(
          alignment.contains(GemmAlignment::M) || (tile_block_rows == THREADGROUP_BLOCK_M),
          [&](auto aligned_m) {
            dispatch_bool(
                alignment.contains(GemmAlignment::N) || (tile_block_cols == THREADGROUP_BLOCK_N),
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
                          d,
                          params->leading_dimension_d,
                          1,
                          epilogue
                      );
                    }
                    accumulator.store_result(d, params->leading_dimension_d);
                  } else {
                    if (needs_epilogue) {
                      accumulator.apply_epilogue_safe(
                          d,
                          params->leading_dimension_d,
                          1,
                          short2(tile_block_cols, tile_block_rows),
                          epilogue
                      );
                    }
                    accumulator.store_result_safe(
                        d,
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

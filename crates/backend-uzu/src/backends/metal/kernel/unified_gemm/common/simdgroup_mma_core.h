#pragma once

#include "../../matmul/common/loader.h"
#include "../../matmul/common/threadgroup_tile.h"
#include "../../generated/matmul.h"
#include "block_geometry.h"

using namespace metal;

namespace uzu {
namespace unified_gemm {

template <
    typename T,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int SIMDGROUPS_PER_ROW,
    int SIMDGROUPS_PER_COLUMN,
    bool MN_ALIGNED,
    bool K_ALIGNED>
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

  template <bool M_aligned, bool N_aligned, bool K_aligned_>
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

    if (!K_aligned_) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      short2 last_tile_dimensions_a = short2(leftover_block_depth, tile_block_rows);
      short2 last_tile_dimensions_b = short2(leftover_block_depth, tile_block_cols);

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
      threadgroup T* a_shared,
      threadgroup T* b_shared,
      uint simd_lane_id,
      uint simd_group_id,
      uint2 threadgroup_position,
      uint3 thread_position
  ) {
    (void)thread_position;

    const uint2 tile_id = swizzled_block_id(threadgroup_position, params->swizzle_log);
    const auto geometry = BlockGeometry<BLOCK_M, BLOCK_N>::compute(tile_id, params);
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
        simd_lane_id);
    thread WeightsLoader loader_b(
        weights,
        params->leading_dimension_b,
        b_shared,
        simd_group_id,
        simd_lane_id);
    thread TileAccumulator accumulator(simd_group_id, simd_lane_id);

    if (MN_ALIGNED) {
      for (int k = 0; k < params->aligned_inner_iterations; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);
        accumulator.multiply_accumulate(a_shared, b_shared);

        loader_a.next();
        loader_b.next();
      }

      threadgroup_barrier(mem_flags::mem_none);

      if (!K_ALIGNED) {
        int leftover_block_depth =
            params->K - params->aligned_inner_iterations * BLOCK_K;
        short2 tile_dimensions_a = short2(leftover_block_depth, BLOCK_M);
        short2 tile_dimensions_b = short2(leftover_block_depth, BLOCK_N);

        loader_a.load_safe(tile_dimensions_a);
        loader_b.load_safe(tile_dimensions_b);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        accumulator.multiply_accumulate(a_shared, b_shared);
      }

      accumulator.store_result(result, params->leading_dimension_d);
      return;
    }

    const ushort tile_block_rows =
        min(BLOCK_M, ((int)params->M) - int(geometry.block_row_start));
    const ushort tile_block_cols =
        min(BLOCK_N, ((int)params->N) - int(geometry.block_col_start));
    const ushort leftover_block_depth =
        params->K - params->aligned_inner_iterations * BLOCK_K;

    if (tile_block_rows == BLOCK_M && tile_block_cols == BLOCK_N) {
      k_loop<true, true, K_ALIGNED>(
          a_shared, b_shared, params->aligned_inner_iterations,
          loader_a, loader_b, accumulator,
          tile_block_rows, tile_block_cols, leftover_block_depth);
      accumulator.store_result(result, params->leading_dimension_d);
    } else if (tile_block_cols == BLOCK_N) {
      k_loop<false, true, K_ALIGNED>(
          a_shared, b_shared, params->aligned_inner_iterations,
          loader_a, loader_b, accumulator,
          tile_block_rows, tile_block_cols, leftover_block_depth);
      accumulator.store_result_safe(
          result,
          params->leading_dimension_d,
          short2(tile_block_cols, tile_block_rows));
    } else if (tile_block_rows == BLOCK_M) {
      k_loop<true, false, K_ALIGNED>(
          a_shared, b_shared, params->aligned_inner_iterations,
          loader_a, loader_b, accumulator,
          tile_block_rows, tile_block_cols, leftover_block_depth);
      accumulator.store_result_safe(
          result,
          params->leading_dimension_d,
          short2(tile_block_cols, tile_block_rows));
    } else {
      k_loop<false, false, K_ALIGNED>(
          a_shared, b_shared, params->aligned_inner_iterations,
          loader_a, loader_b, accumulator,
          tile_block_rows, tile_block_cols, leftover_block_depth);
      accumulator.store_result_safe(
          result,
          params->leading_dimension_d,
          short2(tile_block_cols, tile_block_rows));
    }
  }
};

} // namespace unified_gemm
} // namespace uzu

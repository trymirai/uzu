#pragma once

#include "../../../common/integral_constant.h"
#include "../../../common/thread_context.h"
#include "../../common/defines.h"
#include "../../common/loader.h"
#include "../../common/threadgroup_tile.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "../../../hadamard_transform/hadamard_transform.h"
#include "block_geometry.h"
#include "gemm_rht.h"
#include "gemm_alignment.h"
#include "gemm_tiling.h"
#include "quant_scale_bias.h"
#include "quant_scale_zero_point.h"
#include "operands.h"
#include "schedules/staged.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename OutputElementType_,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    typename LeftOperand,
    typename RightOperand>
struct SimdgroupMmaCore {
  using OutputElementType = OutputElementType_;
  using Left = operands::LeftBinding<LeftOperand, RightOperand>;
  using Right = operands::RightBinding<RightOperand>;
  using LeftElementType = typename Left::ElementType;
  using RightElementType = typename Right::ElementType;
  using LeftArgs = typename Left::Args;
  using RightArgs = typename Right::Args;
  static_assert(
      metal::is_same<LeftOperand, operands::Dense<LeftElementType>>::value,
      "simdgroup MMA stages full-precision left operands only"
  );
  UZU_CONST bool TRANSPOSE_RIGHT = TRANSPOSE_B;
  UZU_CONST int THREADGROUP_BLOCK_M = gemm_tiling_block_m(GEMM_TILING);
  UZU_CONST int THREADGROUP_BLOCK_N = gemm_tiling_block_n(GEMM_TILING);
  UZU_CONST int THREADGROUP_BLOCK_K = gemm_tiling_block_k(GEMM_TILING);
  UZU_CONST int SIMDGROUPS_PER_ROW = gemm_tiling_simdgroups_per_row(GEMM_TILING);
  UZU_CONST int SIMDGROUPS_PER_COLUMN = gemm_tiling_simdgroups_per_column(GEMM_TILING);
  UZU_CONST ushort PADDING_A = 16 / sizeof(LeftElementType);
  UZU_CONST ushort PADDING_B = 16 / sizeof(RightElementType);
  UZU_CONST ushort SHARED_STRIDE_A = THREADGROUP_BLOCK_K + PADDING_A;
  UZU_CONST ushort SHARED_STRIDE_B = (TRANSPOSE_B ? THREADGROUP_BLOCK_K : THREADGROUP_BLOCK_N) + PADDING_B;
  UZU_CONST ushort STAGING_ELEMENTS =
      TRANSPOSE_B ? THREADGROUP_BLOCK_N * SHARED_STRIDE_B : THREADGROUP_BLOCK_K * SHARED_STRIDE_B;
  UZU_CONST ushort THREADGROUP_THREADS = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;

  using ALoader = uzu::matmul::ThreadgroupLoader<
      LeftElementType,
      THREADGROUP_BLOCK_M,
      THREADGROUP_BLOCK_K,
      SHARED_STRIDE_A,
      true,
      THREADGROUP_THREADS>;
  using FullPrecisionRightLoader = uzu::matmul::ThreadgroupLoader<
      RightElementType,
      TRANSPOSE_B ? THREADGROUP_BLOCK_N : THREADGROUP_BLOCK_K,
      TRANSPOSE_B ? THREADGROUP_BLOCK_K : THREADGROUP_BLOCK_N,
      SHARED_STRIDE_B,
      TRANSPOSE_B,
      THREADGROUP_THREADS>;
  using TileAccumulator = uzu::matmul::ThreadgroupTile<
      LeftElementType,
      RightElementType,
      OutputElementType,
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
      uzu::matmul::TransformNone<OutputElementType, float>>;

  template <uint GEMM_ALIGNMENT_RAW, typename BLoader>
  static METAL_FUNC void k_loop(
      threadgroup LeftElementType* a_shared,
      threadgroup RightElementType* b_shared,
      const int aligned_k_iterations,
      thread ALoader& loader_a,
      thread BLoader& loader_b,
      thread TileAccumulator& accumulator,
      thread const ushort& tile_block_rows,
      thread const ushort& tile_block_cols,
      thread const ushort& leftover_block_depth
  ) {
    constexpr GemmAlignment gemm_alignment{GEMM_ALIGNMENT_RAW};
    short2 tile_dimensions_a = short2(THREADGROUP_BLOCK_K, tile_block_rows);
    short2 tile_dimensions_b =
        TRANSPOSE_B ? short2(THREADGROUP_BLOCK_K, tile_block_cols) : short2(tile_block_cols, THREADGROUP_BLOCK_K);

    for (int k = 0; k < aligned_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if constexpr (gemm_alignment.contains(GemmAlignment::M)) {
        loader_a.load_unsafe();
      } else {
        loader_a.load_safe(tile_dimensions_a);
      }
      if constexpr (gemm_alignment.contains(GemmAlignment::N)) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dimensions_b);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      accumulator.matmul(a_shared, b_shared);

      loader_a.next();
      loader_b.next();
    }

    if constexpr (!gemm_alignment.contains(GemmAlignment::K)) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      short2 last_tile_dimensions_a = short2(leftover_block_depth, tile_block_rows);
      short2 last_tile_dimensions_b =
          TRANSPOSE_B ? short2(leftover_block_depth, tile_block_cols) : short2(tile_block_cols, leftover_block_depth);

      loader_a.load_safe(last_tile_dimensions_a);
      loader_b.load_safe(last_tile_dimensions_b);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      accumulator.matmul(a_shared, b_shared);
    }
  }

  template <uint GEMM_ALIGNMENT_RAW>
  static METAL_FUNC void finalize(
      thread TileAccumulator& accumulator,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      const thread ushort& tile_block_rows,
      const thread ushort& tile_block_cols,
      const bool needs_epilogue,
      const thread uzu::matmul::TransformScaleAccumulate<float, float>& epilogue,
      const device RightElementType* bias_block,
      const bool needs_bias,
      const device int32_t* rht_factors_block,
      const bool needs_rht,
      const thread ThreadContext& thread_context
  ) {
    constexpr GemmAlignment gemm_alignment{GEMM_ALIGNMENT_RAW};
    if constexpr (gemm_alignment.contains(GemmAlignment::M) && gemm_alignment.contains(GemmAlignment::N)) {
      if (needs_epilogue) {
        accumulator.apply_epilogue(d, params->leading_dimension_d, 1, epilogue);
      }
      if (needs_bias) {
        accumulator.apply_bias(bias_block);
      }
      accumulator.store_result(d, params->leading_dimension_d);
    } else {
      if (needs_epilogue) {
        accumulator
            .apply_epilogue_safe(d, params->leading_dimension_d, 1, short2(tile_block_cols, tile_block_rows), epilogue);
      }
      if (needs_bias) {
        accumulator.apply_bias_safe(bias_block, short2(tile_block_cols, tile_block_rows));
      }
      accumulator.store_result_safe(d, params->leading_dimension_d, short2(tile_block_cols, tile_block_rows));
    }

    if (needs_rht) {
      threadgroup_barrier(mem_flags::mem_device);
      apply_output_random_hadamard_transform(
          d,
          rht_factors_block,
          tile_block_rows,
          tile_block_cols,
          params->leading_dimension_d,
          ushort(SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN),
          thread_context
      );
    }
  }

  static METAL_FUNC void run(
      LeftArgs left,
      RightArgs right,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      GemmAlignment alignment,
      GemmDTransform output_transform,
      const device RightElementType* output_bias,
      const device int32_t* rht_factors,
      threadgroup LeftElementType* a_shared,
      threadgroup RightElementType* b_shared,
      const thread ThreadContext& thread_context
  ) {
    const uint partition = thread_context.threadgroup_position.z;
    const uint tile_row = thread_context.threadgroup_position.y;
    const uint2 tile = tile_id(uint2(thread_context.threadgroup_position.x, tile_row), params);
    const auto geometry = ThreadgroupTileGeometry<THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N>::compute(tile, params);
    if (geometry.out_of_bounds) {
      return;
    }

    const uint k_offset = partition * params->aligned_inner_iterations * THREADGROUP_BLOCK_K;

    threadgroup_barrier(mem_flags::mem_none);

    const size_t block_row = size_t(geometry.block_row_start);
    const size_t block_col = size_t(geometry.block_col_start);

    left.seek_rows(block_row, params->leading_dimension_a, k_offset);
    d +=
        size_t(partition) * size_t(params->M) * size_t(params->N) + block_row * params->leading_dimension_d + block_col;

    thread ALoader loader_a(left.values, params->leading_dimension_a, a_shared, thread_context);
    thread TileAccumulator accumulator(thread_context);

    const ushort tile_block_rows =
        min(THREADGROUP_BLOCK_M, static_cast<int>(params->M) - static_cast<int>(geometry.block_row_start));
    const ushort tile_block_cols =
        min(THREADGROUP_BLOCK_N, static_cast<int>(params->N) - static_cast<int>(geometry.block_col_start));
    const ushort leftover_block_depth = params->K - params->aligned_inner_iterations * THREADGROUP_BLOCK_K;

    const bool needs_scale = output_transform.contains(GemmDTransform::SCALE);
    const bool needs_accumulate = output_transform.contains(GemmDTransform::ACCUMULATE);
    const bool needs_bias = output_transform.contains(GemmDTransform::BIAS);
    const bool needs_rht = output_transform.contains(GemmDTransform::RHT);
    const bool needs_epilogue = needs_scale || needs_accumulate;
    const float alpha = needs_scale ? params->ab_scale : 1.0f;
    const float beta = needs_accumulate ? 1.0f : 0.0f;
    uzu::matmul::TransformScaleAccumulate<float, float> epilogue(alpha, beta);
    const device RightElementType* bias_block = output_bias + block_col;
    const device int32_t* rht_factors_block = rht_factors + block_col;

    auto loader_b = schedules::StagedRightLoader<RightOperand>::template make<SimdgroupMmaCore>(
        right,
        params,
        block_col,
        k_offset,
        b_shared,
        thread_context
    );

    const bool all_aligned = ((alignment.contains(GemmAlignment::M)) || (tile_block_rows == THREADGROUP_BLOCK_M)) &&
                             ((alignment.contains(GemmAlignment::N)) || (tile_block_cols == THREADGROUP_BLOCK_N)) &&
                             alignment.contains(GemmAlignment::K);
    constexpr uint MASK_ALL =
        static_cast<uint>(GemmAlignment::M) | static_cast<uint>(GemmAlignment::N) | static_cast<uint>(GemmAlignment::K);
    const uint dynamic_alignment_mask =
        alignment.raw_value | ((tile_block_rows == THREADGROUP_BLOCK_M) ? static_cast<uint>(GemmAlignment::M) : 0u) |
        ((tile_block_cols == THREADGROUP_BLOCK_N) ? static_cast<uint>(GemmAlignment::N) : 0u);

    auto kernel_invoke = [&](auto gemm_alignment_mask) {
      constexpr uint gemm_alignment = gemm_alignment_mask.value;
      k_loop<gemm_alignment>(
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
      finalize<gemm_alignment>(
          accumulator,
          d,
          params,
          tile_block_rows,
          tile_block_cols,
          needs_epilogue,
          epilogue,
          bias_block,
          needs_bias,
          rht_factors_block,
          needs_rht,
          thread_context
      );
    };

    if (all_aligned) {
      kernel_invoke(integral_constant<uint, MASK_ALL>{});
    } else {
      dispatch_gemm_alignment(dynamic_alignment_mask, kernel_invoke);
    }
  }
};

} // namespace gemm
} // namespace uzu

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

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename AT,
    typename BT,
    typename DT,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    GemmBPrologueKind B_PROLOGUE = GemmBPrologueKind::FullPrecision,
    int BITS = 0,
    int GROUP_SIZE = 0>
struct SimdgroupMmaCore {
  METAL_CONST int THREADGROUP_BLOCK_M = gemm_tiling_block_m(GEMM_TILING);
  METAL_CONST int THREADGROUP_BLOCK_N = gemm_tiling_block_n(GEMM_TILING);
  METAL_CONST int THREADGROUP_BLOCK_K = gemm_tiling_block_k(GEMM_TILING);
  METAL_CONST int SIMDGROUPS_PER_ROW = gemm_tiling_simdgroups_per_row(GEMM_TILING);
  METAL_CONST int SIMDGROUPS_PER_COLUMN = gemm_tiling_simdgroups_per_column(GEMM_TILING);
  METAL_CONST ushort PADDING_A = 16 / sizeof(AT);
  METAL_CONST ushort PADDING_B = 16 / sizeof(BT);
  METAL_CONST ushort SHARED_STRIDE_A = THREADGROUP_BLOCK_K + PADDING_A;
  METAL_CONST ushort SHARED_STRIDE_B = (TRANSPOSE_B ? THREADGROUP_BLOCK_K : THREADGROUP_BLOCK_N) + PADDING_B;
  METAL_CONST ushort THREADGROUP_THREADS = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;

  using ALoader = uzu::matmul::
      ThreadgroupLoader<AT, THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_K, SHARED_STRIDE_A, true, THREADGROUP_THREADS>;
  using BLoaderFp = uzu::matmul::ThreadgroupLoader<
      BT,
      TRANSPOSE_B ? THREADGROUP_BLOCK_N : THREADGROUP_BLOCK_K,
      TRANSPOSE_B ? THREADGROUP_BLOCK_K : THREADGROUP_BLOCK_N,
      SHARED_STRIDE_B,
      TRANSPOSE_B,
      THREADGROUP_THREADS>;
  using BLoaderScaleBias = QuantizedBlockLoaderScaleBias<
      BT,
      THREADGROUP_BLOCK_N,
      THREADGROUP_BLOCK_K,
      SHARED_STRIDE_B,
      1,
      THREADGROUP_THREADS,
      GROUP_SIZE,
      BITS>;
  using BLoaderScaleZeroPoint = QuantizedBlockLoaderScaleZeroPoint<
      BT,
      THREADGROUP_BLOCK_N,
      THREADGROUP_BLOCK_K,
      SHARED_STRIDE_B,
      1,
      THREADGROUP_THREADS,
      GROUP_SIZE,
      BITS>;
  using BLoaderScaleSymmetric = QuantizedBlockLoaderScaleZeroPoint<
      BT,
      THREADGROUP_BLOCK_N,
      THREADGROUP_BLOCK_K,
      SHARED_STRIDE_B,
      1,
      THREADGROUP_THREADS,
      GROUP_SIZE,
      BITS,
      true>;
  using TileAccumulator = uzu::matmul::ThreadgroupTile<
      AT,
      BT,
      DT,
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
      uzu::matmul::TransformNone<DT, float>>;

  template <uint GEMM_ALIGNMENT_RAW, typename BLoader>
  static METAL_FUNC void k_loop(
      threadgroup AT* a_shared,
      threadgroup BT* b_shared,
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
      accumulator.multiply_accumulate(a_shared, b_shared);

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
      accumulator.multiply_accumulate(a_shared, b_shared);
    }
  }

  template <uint GEMM_ALIGNMENT_RAW>
  static METAL_FUNC void finalize(
      thread TileAccumulator& accumulator,
      device DT* d,
      const constant uzu::matmul::GemmParams* params,
      const thread ushort& tile_block_rows,
      const thread ushort& tile_block_cols,
      const bool needs_epilogue,
      const thread uzu::matmul::TransformScaleAccumulate<float, float>& epilogue,
      const device BT* bias_block,
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
      const device AT* a,
      const device BT* b,
      device DT* d,
      const constant uzu::matmul::GemmParams* params,
      GemmAlignment alignment,
      GemmDTransform output_transform,
      const device BT* scales,
      const device BT* biases,
      const device uint8_t* zero_points,
      const device BT* output_bias,
      const device int32_t* rht_factors,
      threadgroup AT* a_shared,
      threadgroup BT* b_shared,
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

    a += block_row * params->leading_dimension_a + k_offset;
    d +=
        size_t(partition) * size_t(params->M) * size_t(params->N) + block_row * params->leading_dimension_d + block_col;

    thread ALoader loader_a(a, params->leading_dimension_a, a_shared, thread_context);
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
    const device BT* bias_block = output_bias + block_col;
    const device int32_t* rht_factors_block = rht_factors + block_col;

    auto loader_b = [&]() {
      if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
        const device BT* b_block_fp = b + (TRANSPOSE_B ? block_col * params->leading_dimension_b : block_col) +
                                      (TRANSPOSE_B ? k_offset : k_offset * params->leading_dimension_b);
        return BLoaderFp(b_block_fp, params->leading_dimension_b, b_shared, thread_context);
      } else {
        constexpr int pack_factor = get_pack_factor<BITS, 8>();
        constexpr int bytes_per_pack = get_bytes_per_pack<BITS>();
        const int k_elements = int(params->K);
        const int weights_row_stride_bytes = k_elements * bytes_per_pack / pack_factor;
        const int groups_per_row = (k_elements + GROUP_SIZE - 1) / GROUP_SIZE;
        const int k_offset_groups = int(k_offset) / GROUP_SIZE;
        const device uint8_t* weights_block = reinterpret_cast<const device uint8_t*>(b) +
                                              block_col * weights_row_stride_bytes +
                                              int(k_offset) * bytes_per_pack / pack_factor;
        const device BT* scales_offset = scales + block_col * groups_per_row + k_offset_groups;

        if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
          const device BT* biases_offset = biases + block_col * groups_per_row + k_offset_groups;
          return BLoaderScaleBias(
              weights_block,
              scales_offset,
              biases_offset,
              k_elements,
              b_shared,
              thread_context.simdgroup_index,
              thread_context.simd_lane_id
          );
        } else if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant) {
          const int zero_point_stride_per_row = (BITS == 4) ? ((groups_per_row + 1) / 2) : groups_per_row;
          const device uint8_t* zero_points_row_start = zero_points + block_col * zero_point_stride_per_row +
                                                        ((BITS == 4) ? (k_offset_groups / 2) : k_offset_groups);
          return BLoaderScaleZeroPoint(
              weights_block,
              scales_offset,
              zero_points_row_start,
              k_elements,
              groups_per_row,
              b_shared,
              thread_context.simdgroup_index,
              thread_context.simd_lane_id
          );
        } else {
          return BLoaderScaleSymmetric(
              weights_block,
              scales_offset,
              nullptr,
              k_elements,
              groups_per_row,
              b_shared,
              thread_context.simdgroup_index,
              thread_context.simd_lane_id
          );
        }
      }
    }();

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

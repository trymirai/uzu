#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_fragment/ops.h"
#include "../gemm_alignment.h"
#include "../operands.h"
#include "../quantized/cache.h"
#include "../quantized/cursor.h"
#include "tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {

namespace schedules {

template <typename LeftOperand, typename RightOperand>
struct IntegerSchedule {
  UZU_CONST bool NEEDS_CORRECTION = RightOperand::NEEDS_CORRECTION;

  template <typename Core, bool ALIGNED_M, bool ALIGNED_N, bool STAGE_WEIGHT_SCALES>
  static METAL_FUNC typename Core::AccumFragment launch(
      typename Core::LeftStorage left_storage,
      typename Core::RightStorage right_storage,
      threadgroup typename Core::RightElementType* shared,
      const constant uzu::matmul::GemmParams* params,
      const TileContext tile,
      const GemmAlignment,
      const thread ThreadContext& thread_context
  ) {
    static_assert(LeftOperand::QUANTIZED && RightOperand::QUANTIZED, "integer schedule requires quantized operands");
    static_assert(LeftOperand::GROUP_SIZE % Core::SIMDGROUP_BLOCK_K == 0, "left groups must contain MMA chunks");
    static_assert(RightOperand::GROUP_SIZE % Core::SIMDGROUP_BLOCK_K == 0, "right groups must contain MMA chunks");
    static_assert(
        LeftOperand::GROUP_SIZE % RightOperand::GROUP_SIZE == 0 ||
            RightOperand::GROUP_SIZE % LeftOperand::GROUP_SIZE == 0,
        "left and right quantization groups must divide one another"
    );
    constexpr ushort SPAN =
        LeftOperand::GROUP_SIZE < RightOperand::GROUP_SIZE ? LeftOperand::GROUP_SIZE : RightOperand::GROUP_SIZE;

    auto left_codes = quantized::make_cursor<quantized::Axis::Rows, Core, typename LeftOperand::Format, ALIGNED_M>(
        left_storage,
        params,
        tile,
        thread_context
    );
    auto right_codes = quantized::make_cursor<quantized::Axis::Columns, Core, typename RightOperand::Format, ALIGNED_N>(
        right_storage,
        params,
        tile,
        thread_context
    );

    quantized::Cache<quantized::Residency::Registers, Core, typename Core::LeftStorage, LeftOperand, ALIGNED_M>
        left_scales(left_storage, shared, params, tile, thread_context);
    quantized::Cache<
        STAGE_WEIGHT_SCALES ? quantized::Residency::Threadgroup : quantized::Residency::Registers,
        Core,
        typename Core::RightStorage,
        RightOperand,
        ALIGNED_N>
        right_scales(right_storage, shared, params, tile, thread_context);

    typename Core::AccumFragment accumulator;
    accumulator.clear();

    if constexpr (STAGE_WEIGHT_SCALES) {
      right_scales.prefetch(0);
    }

    const int k_group_count = int(params->aligned_inner_iterations);
    constexpr int chunks_per_span = int(SPAN) / Core::SIMDGROUP_BLOCK_K;
    constexpr int spans_per_k_group = int(RightOperand::GROUP_SIZE) / int(SPAN);

    METAL_PRAGMA_NO_UNROLL
    for (int k_group_index = 0; k_group_index < k_group_count; ++k_group_index) {
      const uint k_group_offset = uint(k_group_index * int(RightOperand::GROUP_SIZE));
      left_codes.begin_k_group(k_group_offset);
      right_codes.begin_k_group(k_group_offset);
      if constexpr (STAGE_WEIGHT_SCALES) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_group_index + 1 < k_group_count) {
          right_scales.prefetch(uint(k_group_index + 1));
        }
      }
      right_scales.fill(k_group_index);

      for (int span_index = 0; span_index < spans_per_k_group; ++span_index) {
        const uint span_offset = uint(k_group_index * int(RightOperand::GROUP_SIZE)) + uint(span_index * int(SPAN));
        left_scales.fill_at_k_offset(span_offset);

        uzu::matmul::Fragment<int, Core::TILES_M, Core::TILES_N, typename Core::FragmentOps> products;
        products.clear();
        METAL_PRAGMA_NO_UNROLL
        for (int chunk = 0; chunk < chunks_per_span; ++chunk) {
          auto left_tile = left_codes.load(uint(chunk));
          auto right_tile = right_codes.load(uint(chunk));
          uzu::matmul::fragment_mma(products, left_tile, right_tile);
          left_codes.advance();
          right_codes.advance();
        }

        Core::AccumFragment::zip_for_each_coord(
            thread_context.simd_lane_id,
            [&](short row, short column, thread float& accumulated, thread int& product) {
              if (!ALIGNED_M && row >= tile.simdgroup_limit_m) {
                return;
              }
              if (!ALIGNED_N && column >= tile.simdgroup_limit_n) {
                return;
              }
              accumulated += left_scales.scale(row) * right_scales.scale(column) * float(product);
            },
            accumulator,
            products
        );

        if constexpr (NEEDS_CORRECTION) {
          accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short column, float value) {
            if (!ALIGNED_M && row >= tile.simdgroup_limit_m) {
              return value;
            }
            if (!ALIGNED_N && column >= tile.simdgroup_limit_n) {
              return value;
            }
            return value + right_scales.correction(column) * left_scales.correction(row);
          });
        }
      }
    }

    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu

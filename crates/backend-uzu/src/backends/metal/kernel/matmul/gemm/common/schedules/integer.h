#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_fragment/ops.h"
#include "../gemm_alignment.h"
#include "../integer_source.h"
#include "../quantized/metadata.h"
#include "../quantized/slice.h"
#include "../quantized/source.h"
#include "tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

template <typename LeftOperand, typename RightOperand>
struct IntegerSchedule {
  using RightFormat = typename RightOperand::Format;
  using RightBinding = operands::RightBinding<RightOperand>;
  UZU_CONST bool needs_correction = RightBinding::NEEDS_CORRECTION;

  template <typename Core, bool ALIGNED_M, bool ALIGNED_N>
  static METAL_FUNC typename Core::AccumFragment launch(
      typename Core::LeftArgs left,
      typename Core::RightArgs right,
      threadgroup typename Core::RightElementType*,
      const constant uzu::matmul::GemmParams* params,
      const TileContext tile,
      const GemmAlignment,
      const thread ThreadContext& thread_context
  ) {
    using Ops = typename Core::FragmentOps;
    using AccumFragment = typename Core::AccumFragment;
    using LeftScaleCache = QuantizationLineCache<
        Core::TILES_M,
        Ops::THREAD_ELEMENT_ROWS,
        Ops::FRAGMENT_ROWS,
        Ops::THREAD_ELEMENT_ROW_STRIDE>;
    using RightScaleCache = QuantizationLineCache<Core::TILES_N, Ops::THREAD_ELEMENT_COLS, Ops::FRAGMENT_COLS, 1>;
    using CorrectionPolicy = Correction<RightOperand>;
    using ColumnCorrectionCache = metal::conditional_t<needs_correction, RightScaleCache, NoCorrectionCache>;
    using RowCorrectionCache = metal::conditional_t<needs_correction, LeftScaleCache, NoCorrectionCache>;

    static_assert(RightOperand::GROUP_SIZE % Core::SIMDGROUP_BLOCK_K == 0, "right groups must contain MMA chunks");

    const auto slice = make_quantized_slice<RightOperand>(tile.k_offset, int(params->K));
    right.storage.seek_block(tile.block_col, tile.k_offset, slice.row_stride_bytes);
    right.storage.seek_columns(tile.tile_col_offset, slice.row_stride_bytes);

    const int leading_dimension_a = int(params->leading_dimension_a);
    const int b_row_stride_bytes = slice.row_stride_bytes;
    const int right_group_iterations = int(params->aligned_inner_iterations);
    const short simdgroup_limit_m = tile.simdgroup_limit_m;
    const short simdgroup_limit_n = tile.simdgroup_limit_n;
    const uint abs_row_base = tile.abs_row_base;
    const uint abs_col_base = tile.absolute_column_base();
    const uint k_offset_right_groups = uint(slice.first_group);
    const uint k_offset_left_scale_groups = tile.k_offset / uint(LeftOperand::GROUP_SIZE);
    const uint right_groups_per_row = uint(slice.groups_per_row);
    const uint left_scale_groups_per_row = uint(params->K) / uint(LeftOperand::GROUP_SIZE);

    AccumFragment accumulator;
    accumulator.clear();

    const short2 position = Ops::get_position(thread_context.simd_lane_id);
    constexpr int k_bytes_per_right_group = int(RightOperand::GROUP_SIZE) * RightOperand::BITS / 8;
    constexpr int left_groups_per_right_group = int(RightOperand::GROUP_SIZE) / int(LeftOperand::GROUP_SIZE);

    METAL_PRAGMA_NO_UNROLL
    for (int right_group = 0; right_group < right_group_iterations; ++right_group) {
      const uint right_group_index = k_offset_right_groups + uint(right_group);

      RightScaleCache right_scales;
      ColumnCorrectionCache column_corrections;
      RightScaleCache::template for_each_line<ALIGNED_N>(
          position.x,
          simdgroup_limit_n,
          abs_col_base,
          right_groups_per_row,
          right_group_index,
          [&](ushort tile_col, ushort thread_col, uint right_scale_index, uint right_column) {
            const float scale = static_cast<float>(right.scales[right_scale_index]);
            right_scales.slot(tile_col, thread_col) = scale;
            if constexpr (needs_correction) {
              column_corrections.slot(tile_col, thread_col) =
                  scale * CorrectionPolicy::midpoint() + CorrectionPolicy::offset(
                                                             scale,
                                                             right_scale_index,
                                                             right_column,
                                                             right_group_index,
                                                             right_groups_per_row,
                                                             right
                                                         );
            }
          }
      );

      RowCorrectionCache row_corrections;
      if constexpr (needs_correction) {
        row_corrections.clear();
      }

      METAL_PRAGMA_NO_UNROLL
      for (int left_group = 0; left_group < left_groups_per_right_group; ++left_group) {
        const int k_element_offset = left_group * int(LeftOperand::GROUP_SIZE);
        auto left_tile = load_integer_tile<int8_t, Ops, ALIGNED_M, Core::TILES_M, Core::TILES_K>(
            left.values + k_element_offset,
            leading_dimension_a,
            simdgroup_limit_m,
            thread_context.simd_lane_id
        );

        uzu::matmul::Fragment<int, Core::TILES_M, Core::TILES_N, Ops> chunk_products;
        auto right_tile =
            QuantizedSource<RightFormat, ALIGNED_N, Core::TILES_N, Core::TILES_K, Core::SIMDGROUP_BLOCK_K>::make(
                right.storage.values,
                k_element_offset,
                b_row_stride_bytes,
                simdgroup_limit_n,
                position,
                thread_context.simd_lane_id
            );
        Ops::template fragment_mm<false, true>(chunk_products, left_tile, right_tile);

        const uint left_scale_group_index =
            k_offset_left_scale_groups + uint(right_group * left_groups_per_right_group + left_group);
        LeftScaleCache left_scales;
        LeftScaleCache::template for_each_line<ALIGNED_M>(
            position.y,
            simdgroup_limit_m,
            abs_row_base,
            left_scale_groups_per_row,
            left_scale_group_index,
            [&](ushort tile_row, ushort thread_row, uint left_scale_index, uint) {
              const float left_scale = left.scales[left_scale_index];
              left_scales.slot(tile_row, thread_row) = left_scale;
              if constexpr (needs_correction) {
                row_corrections.slot(tile_row, thread_row) += left_scale * float(left.group_sums[left_scale_index]);
              }
            }
        );

        AccumFragment::zip_for_each_coord(
            thread_context.simd_lane_id,
            [&](short row, short col, thread float& accumulated, thread int& products) {
              if (!ALIGNED_M && row >= simdgroup_limit_m) {
                return;
              }
              if (!ALIGNED_N && col >= simdgroup_limit_n) {
                return;
              }
              accumulated += left_scales.at(row - position.y) * right_scales.at(col - position.x) * float(products);
            },
            accumulator,
            chunk_products
        );
      }

      if constexpr (needs_correction) {
        accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, float value) {
          if (!ALIGNED_M && row >= simdgroup_limit_m) {
            return value;
          }
          if (!ALIGNED_N && col >= simdgroup_limit_n) {
            return value;
          }
          return value + column_corrections.at(col - position.x) * row_corrections.at(row - position.y);
        });
      }

      left.advance_k(RightOperand::GROUP_SIZE);
      right.storage.advance_k(k_bytes_per_right_group);
    }

    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu

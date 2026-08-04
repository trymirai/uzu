#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_fragment/ops.h"
#include "../integer_source.h"
#include "../quantized/metadata.h"
#include "../quantized/source.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

template <ushort BITS_VALUE, ushort RIGHT_GROUP_SIZE_VALUE, ushort LEFT_GROUP_SIZE_VALUE, GemmBPrologueKind B_PROLOGUE>
struct IntegerSchedule {
  using RightFormat = uzu::matmul::IntegerFormat<BITS_VALUE, uzu::matmul::Signedness::Signed>;
  UZU_CONST ushort RIGHT_GROUP_SIZE = RIGHT_GROUP_SIZE_VALUE;
  UZU_CONST ushort LEFT_GROUP_SIZE = LEFT_GROUP_SIZE_VALUE;
  UZU_CONST bool needs_correction = B_PROLOGUE != GemmBPrologueKind::ScaleSymmetricDequant;

  template <typename Core, bool ALIGNED_M, bool ALIGNED_N>
  static METAL_FUNC typename Core::AccumFragment run(
      const device int8_t* left_integer_simdgroup,
      const device uint8_t* right_packed_simdgroup,
      const device float* left_scale_values,
      const device int32_t* left_group_sums,
      const device typename Core::RightType* right_scale_values,
      const device typename Core::RightType* biases,
      const device uint8_t* zero_points,
      const int leading_dimension_a,
      const int b_row_stride_bytes,
      const int right_group_iterations,
      const short simdgroup_limit_m,
      const short simdgroup_limit_n,
      const uint abs_row_base,
      const uint abs_col_base,
      const uint k_offset_right_groups,
      const uint k_offset_left_scale_groups,
      const uint right_groups_per_row,
      const uint left_scale_groups_per_row,
      const thread ThreadContext& thread_context
  ) {
    using Ops = typename Core::FragmentOps;
    using AccumFragment = typename Core::AccumFragment;
    using LeftScaleCache = QuantizationLineCache<
        Core::TILES_M,
        Ops::THREAD_ELEMENT_ROWS,
        Ops::FRAGMENT_ROWS,
        Ops::THREAD_ELEMENT_ROW_STRIDE,
        LEFT_GROUP_SIZE>;
    using RightScaleCache =
        QuantizationLineCache<Core::TILES_N, Ops::THREAD_ELEMENT_COLS, Ops::FRAGMENT_COLS, 1, RIGHT_GROUP_SIZE>;
    using CorrectionPolicy = Correction<B_PROLOGUE, RightFormat, typename Core::RightType>;
    using ColumnCorrectionCache = metal::conditional_t<needs_correction, RightScaleCache, NoCorrectionCache>;
    using RowCorrectionCache = metal::conditional_t<needs_correction, LeftScaleCache, NoCorrectionCache>;

    static_assert(RIGHT_GROUP_SIZE % LEFT_GROUP_SIZE == 0, "right groups must contain left groups");
    static_assert(RIGHT_GROUP_SIZE % Core::SIMDGROUP_BLOCK_K == 0, "right groups must contain MMA chunks");

    AccumFragment accumulator;
    accumulator.clear();

    const short2 position = Ops::get_position(thread_context.simd_lane_id);
    constexpr int k_bytes_per_right_group = int(RIGHT_GROUP_SIZE) * BITS_VALUE / 8;
    constexpr int left_groups_per_right_group = int(RIGHT_GROUP_SIZE) / int(LEFT_GROUP_SIZE);

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
            const float scale = static_cast<float>(right_scale_values[right_scale_index]);
            right_scales.slot(tile_col, thread_col) = scale;
            if constexpr (needs_correction) {
              column_corrections.slot(tile_col, thread_col) =
                  scale * CorrectionPolicy::midpoint() + CorrectionPolicy::offset(
                                                             scale,
                                                             right_scale_index,
                                                             right_column,
                                                             right_group_index,
                                                             right_groups_per_row,
                                                             biases,
                                                             zero_points
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
        const int k_element_offset = left_group * int(LEFT_GROUP_SIZE);
        auto left_tile = load_integer_tile<int8_t, Ops, ALIGNED_M, Core::TILES_M, Core::TILES_K>(
            left_integer_simdgroup + k_element_offset,
            leading_dimension_a,
            simdgroup_limit_m,
            thread_context.simd_lane_id
        );

        uzu::matmul::Fragment<int, Core::TILES_M, Core::TILES_N, Ops> chunk_products;
        auto right =
            QuantizedSource<RightFormat, ALIGNED_N, Core::TILES_N, Core::TILES_K, Core::SIMDGROUP_BLOCK_K>::make(
                right_packed_simdgroup,
                k_element_offset,
                b_row_stride_bytes,
                simdgroup_limit_n,
                position,
                thread_context.simd_lane_id
            );
        Ops::template fragment_mm<false, true>(chunk_products, left_tile, right);

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
              const float left_scale = left_scale_values[left_scale_index];
              left_scales.slot(tile_row, thread_row) = left_scale;
              if constexpr (needs_correction) {
                row_corrections.slot(tile_row, thread_row) += left_scale * float(left_group_sums[left_scale_index]);
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

      left_integer_simdgroup += RIGHT_GROUP_SIZE;
      right_packed_simdgroup += k_bytes_per_right_group;
    }

    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu

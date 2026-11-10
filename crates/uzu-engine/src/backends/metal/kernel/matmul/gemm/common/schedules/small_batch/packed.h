#pragma once

#include <metal_stdlib>

#include "../../../../../generated/gemm.h"
#include "../../../../common/fragment.h"
#include "../../../../layout/interleaved_rows_layout.h"
#include "../../../../common/quant_unpack.h"
#include "../../../../common/simdgroup_fragment_ops.h"
#include "../../../../common/packed_half_layout.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

struct PackedSchedule {
  UZU_CONST bool TRANSPOSED_OUTPUT = true;

  template <ushort BITS, ushort WORDS_PER_LANE>
  struct WeightDecoder {
    thread const uint* packed_words;

    METAL_FUNC half2 operator()(const ushort row, const ushort k) const thread {
      if constexpr (BITS == 2) {
        return w2_pair(packed_words[row * WORDS_PER_LANE + k / 8], k % 8);
      } else {
        return w4_pair(packed_words[row * WORDS_PER_LANE + k / 4], k % 4);
      }
    }
  };

  template <typename Core>
  static METAL_FUNC typename Core::AccumFragment launch(
      const typename Core::LeftStorage left,
      const typename Core::RightStorage right,
      const constant uzu::matmul::GemmParams* params,
      const thread typename Core::TileContext& tile_context
  ) {
    using RightOperand = typename Core::Right;
    constexpr bool HAS_BIAS = RightOperand::SCHEME == GemmBPrologueKind::ScaleBiasDequant;
    constexpr ushort WEIGHT_GROUP_SIZE = RightOperand::GROUP_SIZE;
    constexpr ushort WEIGHT_BITS = RightOperand::BITS;
    constexpr ushort K_STEPS_PER_GROUP = PackedHalfLayout<WEIGHT_GROUP_SIZE, WEIGHT_BITS>::STEPS;
    constexpr ushort WORDS_PER_LANE = WEIGHT_GROUP_SIZE * WEIGHT_BITS / (8 * sizeof(uint4));
    constexpr ushort WORDS_PER_GROUP = PackedHalfLayout<WEIGHT_GROUP_SIZE, WEIGHT_BITS>::GROUP_WORDS;
    constexpr ushort GROUPS_PER_K_BLOCK = Core::THREADGROUP_BLOCK_K / WEIGHT_GROUP_SIZE;
    using Ops = typename Core::FragmentOps;
    using WeightFragment = uzu::matmul::
        GeneratedFragment<half, Core::TILES_M, K_STEPS_PER_GROUP, Ops, WeightDecoder<WEIGHT_BITS, WORDS_PER_LANE>>;
    using ActivationFragment = uzu::matmul::Fragment<half, K_STEPS_PER_GROUP, Core::TILES_N, Ops>;

    const device uint* packed_activations = left.packed_half;
    const uint fragment_column_pair = uint(tile_context.lane_position.x) / 2;
    const uint lane_row = uint(tile_context.lane_position.y);
    const uint weight_sign_mask = right.sign_mask();

    const bool interleaved_rows = right.gemm_b_layout == GemmBLayout::InterleavedRows;
    const bool row_scaled = right.hoist_row_metadata;
    typename Core::AccumFragment accumulator;
    accumulator.clear();
    typename Core::AccumFragment group_product;
    group_product.clear();
    float4 total_sums = 0.0f;
    const uint k_block_end = tile_context.k_block_end;
    for (uint k_block = tile_context.k_block_begin + tile_context.simdgroup_k_slice; k_block < k_block_end;
         k_block += Core::SIMDGROUPS_PER_K) {
      const uint first_weight_group = k_block * GROUPS_PER_K_BLOCK;
      METAL_PRAGMA_UNROLL
      for (ushort group_in_block = 0; group_in_block < GROUPS_PER_K_BLOCK; ++group_in_block) {
        const uint weight_group_index = first_weight_group + group_in_block;
        const device uint* packed_group = packed_activations + weight_group_index * WORDS_PER_GROUP;
        const device uint* activation_pairs =
            packed_group +
            PackedHalfLayout<WEIGHT_GROUP_SIZE, WEIGHT_BITS>::pair_offset(fragment_column_pair, lane_row);
        ActivationFragment activation_fragment;
        METAL_PRAGMA_UNROLL
        for (ushort vector = 0; vector < K_STEPS_PER_GROUP / 4; ++vector) {
          const uint4 raw = reinterpret_cast<const device uint4*>(activation_pairs)[vector];
          activation_fragment.fragment_at(4 * vector, 0) = as_type<half2>(raw.x);
          activation_fragment.fragment_at(4 * vector + 1, 0) = as_type<half2>(raw.y);
          activation_fragment.fragment_at(4 * vector + 2, 0) = as_type<half2>(raw.z);
          activation_fragment.fragment_at(4 * vector + 3, 0) = as_type<half2>(raw.w);
        }
        const float4 activation_sums = reinterpret_cast<const device float4*>(
            packed_group + PackedHalfLayout<WEIGHT_GROUP_SIZE, WEIGHT_BITS>::sums_offset()
        )[fragment_column_pair];

        uint weight_rows[Core::TILES_M];
        float weight_scales[Core::TILES_M];
        float weight_offsets[Core::TILES_M]; // zero point, or bias
        uint packed_weight_words[Core::TILES_M * WORDS_PER_LANE];
        uint4 word_runs[WORDS_PER_LANE];
        if (interleaved_rows) {
          METAL_PRAGMA_UNROLL
          for (ushort word_index = 0; word_index < WORDS_PER_LANE; ++word_index) {
            word_runs[word_index] = InterleavedRowsLayout::lane_run(
                right.codes,
                tile_context.simdgroup_column_base + lane_row,
                weight_group_index * (WEIGHT_GROUP_SIZE / W4_CODES_PER_WORD) + fragment_column_pair * WORDS_PER_LANE +
                    word_index,
                params->K / W4_CODES_PER_WORD
            );
          }
        }
        METAL_PRAGMA_UNROLL
        for (ushort tile_m = 0; tile_m < Core::TILES_M; ++tile_m) {
          weight_rows[tile_m] =
              min(tile_context.simdgroup_column_base + Ops::FRAGMENT_ROWS * tile_m + lane_row, params->N - 1);
          const device uint* weight_source = reinterpret_cast<const device uint*>(
              right.row_codes(weight_rows[tile_m], params) +
              PackedHalfLayout<WEIGHT_GROUP_SIZE, WEIGHT_BITS>::codes_offset(weight_group_index, fragment_column_pair)
          );
          METAL_PRAGMA_UNROLL
          for (ushort word_index = 0; word_index < WORDS_PER_LANE; ++word_index) {
            const uint word = interleaved_rows ? word_runs[word_index][tile_m] : weight_source[word_index];
            packed_weight_words[tile_m * WORDS_PER_LANE + word_index] = word ^ weight_sign_mask;
          }
          if (!row_scaled) {
            weight_scales[tile_m] = float(right.scale(weight_rows[tile_m], weight_group_index, params));
            if constexpr (HAS_BIAS) {
              weight_offsets[tile_m] =
                  float(right.bias()[right.scale_index(weight_rows[tile_m], weight_group_index, params)]);
            } else {
              weight_offsets[tile_m] = float(right.zero_point(weight_rows[tile_m], weight_group_index, params));
            }
          }
        }

        WeightDecoder<WEIGHT_BITS, WORDS_PER_LANE> weight_decoder{packed_weight_words};
        WeightFragment weight_fragment{weight_decoder};
        if (row_scaled) {
          uzu::matmul::fragment_mma(group_product, weight_fragment, activation_fragment);
          total_sums += activation_sums;
          continue;
        }
        group_product.clear();
        uzu::matmul::fragment_mma(group_product, weight_fragment, activation_fragment);
        accumulator.for_each_lane(
            [&](ushort tile_m, ushort input_index, thread float& accumulator_value, thread float& group_product_value) {
              if constexpr (HAS_BIAS) {
                accumulator_value =
                    fma(group_product_value - activation_sums[2 * input_index + 1],
                        weight_scales[tile_m],
                        fma(weight_offsets[tile_m], activation_sums[2 * input_index], accumulator_value));
              } else {
                const float correction =
                    fma(weight_offsets[tile_m], activation_sums[2 * input_index], activation_sums[2 * input_index + 1]);
                accumulator_value = fma(group_product_value - correction, weight_scales[tile_m], accumulator_value);
              }
            },
            group_product
        );
      }
    }

    if constexpr (HAS_BIAS) {
      if (row_scaled) {
        accumulator.for_each_lane(
            [&](ushort tile_m, ushort input_index, thread float& accumulator_value, thread float& product_value) {
              const uint row =
                  min(tile_context.simdgroup_column_base + Ops::FRAGMENT_ROWS * tile_m + lane_row, params->N - 1);
              accumulator_value =
                  fma(product_value - total_sums[2 * input_index + 1],
                      float(right.scale(row, 0, params)),
                      float(right.bias()[right.scale_index(row, 0u, params)]) * total_sums[2 * input_index]);
            },
            group_product
        );
      }
    }

    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu

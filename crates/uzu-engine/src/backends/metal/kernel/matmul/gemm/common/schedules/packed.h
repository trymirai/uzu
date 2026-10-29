#pragma once

#include <metal_stdlib>

#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/quant_unpack.h"
#include "../../../common/simdgroup_fragment_ops.h"
#include "../packed_half_layout.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

static METAL_FUNC half2 w4_pair(uint word, ushort nibble) {
  switch (nibble) {
  case 0:
    return as_type<half2>((word & 0x000f000fu) | 0x64006400u);
  case 1:
    return as_type<half2>((word & 0x00f000f0u) | 0x54005400u);
  case 2:
    return as_type<half2>(((word >> 6) & 0x003c003cu) | 0x5c005c00u);
  default:
    return as_type<half2>(((word >> 6) & 0x03c003c0u) | 0x4c004c00u);
  }
}

struct PackedSchedule {
  UZU_CONST bool TRANSPOSED_OUTPUT = true;

  template <ushort WORDS_PER_LANE>
  struct WeightDecoder {
    thread const uint* packed_words;

    METAL_FUNC half2 operator()(const ushort row, const ushort k) const thread {
      return w4_pair(packed_words[row * WORDS_PER_LANE + k / 4], k % 4);
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
    static_assert(RightOperand::QUANTIZED && RightOperand::BITS == 4, "packed schedule requires W4 weights");
    constexpr ushort WEIGHT_GROUP_SIZE = RightOperand::GROUP_SIZE;
    constexpr ushort K_STEPS_PER_GROUP = PackedHalfLayout<WEIGHT_GROUP_SIZE>::STEPS;
    constexpr ushort WORDS_PER_LANE = WEIGHT_GROUP_SIZE / 32;
    constexpr ushort WORDS_PER_GROUP = PackedHalfLayout<WEIGHT_GROUP_SIZE>::GROUP_WORDS;
    constexpr ushort GROUPS_PER_K_BLOCK = Core::THREADGROUP_BLOCK_K / WEIGHT_GROUP_SIZE;
    using Ops = typename Core::FragmentOps;
    using WeightFragment =
        uzu::matmul::GeneratedFragment<half, Core::TILES_M, K_STEPS_PER_GROUP, Ops, WeightDecoder<WORDS_PER_LANE>>;
    using ActivationFragment = uzu::matmul::Fragment<half, K_STEPS_PER_GROUP, Core::TILES_N, Ops>;

    const device uint* packed_activations = left.packed_half;
    const uint fragment_column_pair = uint(tile_context.lane_position.x) / 2;
    const uint lane_row = uint(tile_context.lane_position.y);
    const uint weight_sign_mask = right.sign_mask();

    typename Core::AccumFragment accumulator;
    accumulator.clear();
    const auto k_blocks = tile_context.k_blocks;
    for (uint k_block = k_blocks.begin + tile_context.simdgroup_k_slice; k_block < k_blocks.end;
         k_block += Core::SIMDGROUPS_PER_K) {
      const uint first_weight_group = k_block * GROUPS_PER_K_BLOCK;
      for (ushort group_in_block = 0; group_in_block < GROUPS_PER_K_BLOCK; ++group_in_block) {
        const uint weight_group_index = first_weight_group + group_in_block;
        const device uint* packed_group = packed_activations + weight_group_index * WORDS_PER_GROUP;
        const device uint* activation_pairs =
            packed_group + PackedHalfLayout<WEIGHT_GROUP_SIZE>::pair_offset(fragment_column_pair, lane_row);
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
            packed_group + PackedHalfLayout<WEIGHT_GROUP_SIZE>::sums_offset()
        )[fragment_column_pair];

        uint weight_rows[Core::TILES_M];
        float weight_scales[Core::TILES_M];
        float weight_zero_points[Core::TILES_M];
        uint packed_weight_words[Core::TILES_M * WORDS_PER_LANE];
        METAL_PRAGMA_UNROLL
        for (ushort tile_m = 0; tile_m < Core::TILES_M; ++tile_m) {
          weight_rows[tile_m] =
              min(tile_context.simdgroup_column_base + Ops::FRAGMENT_ROWS * tile_m + lane_row, params->N - 1);
          const device uint* weight_source = reinterpret_cast<const device uint*>(
              right.row_codes(weight_rows[tile_m], params) +
              PackedHalfLayout<WEIGHT_GROUP_SIZE>::codes_offset(weight_group_index, fragment_column_pair)
          );
          METAL_PRAGMA_UNROLL
          for (ushort word_index = 0; word_index < WORDS_PER_LANE; ++word_index) {
            packed_weight_words[tile_m * WORDS_PER_LANE + word_index] = weight_source[word_index] ^ weight_sign_mask;
          }
          weight_scales[tile_m] = float(right.scale(weight_rows[tile_m], weight_group_index, params));
          weight_zero_points[tile_m] = float(right.zero_point(weight_rows[tile_m], weight_group_index, params));
        }

        WeightDecoder<WORDS_PER_LANE> weight_decoder{packed_weight_words};
        WeightFragment weight_fragment{weight_decoder};
        typename Core::AccumFragment group_product;
        group_product.clear();
        uzu::matmul::fragment_mma(group_product, weight_fragment, activation_fragment);
        accumulator.for_each_lane(
            [&](ushort tile_m, ushort input_index, thread float& accumulator_value, thread float& group_product_value) {
              const float correction =
                  fma(weight_zero_points[tile_m],
                      activation_sums[2 * input_index],
                      activation_sums[2 * input_index + 1]);
              accumulator_value = fma(group_product_value - correction, weight_scales[tile_m], accumulator_value);
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

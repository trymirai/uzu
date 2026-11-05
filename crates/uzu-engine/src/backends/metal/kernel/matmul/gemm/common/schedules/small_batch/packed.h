#pragma once

#include <metal_stdlib>

#include "../../../../../generated/gemm.h"
#include "../../../../common/fragment.h"
#include "../../../../common/quant_unpack.h"
#include "../../../../common/simdgroup_fragment_ops.h"
#include "../../../../common/packed_half_layout.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

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
    constexpr bool HAS_BIAS = RightOperand::SCHEME == GemmBPrologueKind::ScaleBiasDequant;
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
    // GemmBLayout::SmallBatch: one unit per (32-row tile, weight group) in the order the lanes consume it (see
    // small_batch_units.rs); other layouts read the production row-major buffers.
    const bool units = right.gemm_b_layout == GemmBLayout::SmallBatch;
    constexpr uint LANE_CODE_BYTES = Core::TILES_M * WORDS_PER_LANE * 4;
    constexpr uint CODE_BYTES = 32 * LANE_CODE_BYTES;
    constexpr uint SCALE_BYTES = 8 * 8;                        // 8 lane rows x 4 tile rows, bf16
    constexpr uint OFFSET_BYTES = HAS_BIAS ? SCALE_BYTES : 16; // bf16 biases, or 4 zero-point nibbles per lane row
    constexpr uint UNIT_BYTES = CODE_BYTES + SCALE_BYTES + OFFSET_BYTES;
    const uint lane_id = tile_context.simd_lane_id;
    const uint tile_n = tile_context.simdgroup_column_base / (Core::TILES_M * Ops::FRAGMENT_ROWS);
    const uint groups_total = params->K / WEIGHT_GROUP_SIZE;

    typename Core::AccumFragment accumulator;
    accumulator.clear();
    const uint k_block_end = tile_context.k_block_end;
    for (uint k_block = tile_context.k_block_begin + tile_context.simdgroup_k_slice; k_block < k_block_end;
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

        uint packed_weight_words[Core::TILES_M * WORDS_PER_LANE];
        float weight_scales[Core::TILES_M];
        float weight_offsets[Core::TILES_M]; // zero point, or bias
        if (units) {
          const device uchar* unit = right.codes + (ulong(tile_n) * groups_total + weight_group_index) * UNIT_BYTES;
          const device uint4* lane_codes = reinterpret_cast<const device uint4*>(unit + lane_id * LANE_CODE_BYTES);
          METAL_PRAGMA_UNROLL
          for (ushort vector = 0; vector < LANE_CODE_BYTES / 16; ++vector) {
            const uint4 raw = lane_codes[vector];
            packed_weight_words[4 * vector] = raw.x ^ weight_sign_mask;
            packed_weight_words[4 * vector + 1] = raw.y ^ weight_sign_mask;
            packed_weight_words[4 * vector + 2] = raw.z ^ weight_sign_mask;
            packed_weight_words[4 * vector + 3] = raw.w ^ weight_sign_mask;
          }
          const uint2 scales = reinterpret_cast<const device uint2*>(unit + CODE_BYTES)[lane_row];
          weight_scales[0] = as_type<float>(scales.x << 16);
          weight_scales[1] = as_type<float>(scales.x & 0xFFFF0000u);
          weight_scales[2] = as_type<float>(scales.y << 16);
          weight_scales[3] = as_type<float>(scales.y & 0xFFFF0000u);
          if constexpr (HAS_BIAS) {
            const uint2 biases = reinterpret_cast<const device uint2*>(unit + CODE_BYTES + SCALE_BYTES)[lane_row];
            weight_offsets[0] = as_type<float>(biases.x << 16);
            weight_offsets[1] = as_type<float>(biases.x & 0xFFFF0000u);
            weight_offsets[2] = as_type<float>(biases.y << 16);
            weight_offsets[3] = as_type<float>(biases.y & 0xFFFF0000u);
          } else {
            const uint zero_points = reinterpret_cast<const device ushort*>(unit + CODE_BYTES + SCALE_BYTES)[lane_row];
            weight_offsets[0] = float(zero_points & 0xFu);
            weight_offsets[1] = float((zero_points >> 4) & 0xFu);
            weight_offsets[2] = float((zero_points >> 8) & 0xFu);
            weight_offsets[3] = float(zero_points >> 12);
          }
        } else {
          METAL_PRAGMA_UNROLL
          for (ushort tile_m = 0; tile_m < Core::TILES_M; ++tile_m) {
            const uint weight_row =
                min(tile_context.simdgroup_column_base + Ops::FRAGMENT_ROWS * tile_m + lane_row, params->N - 1);
            const device uint* weight_source = reinterpret_cast<const device uint*>(
                right.row_codes(weight_row, params) +
                PackedHalfLayout<WEIGHT_GROUP_SIZE>::codes_offset(weight_group_index, fragment_column_pair)
            );
            METAL_PRAGMA_UNROLL
            for (ushort word_index = 0; word_index < WORDS_PER_LANE; ++word_index) {
              packed_weight_words[tile_m * WORDS_PER_LANE + word_index] = weight_source[word_index] ^ weight_sign_mask;
            }
            weight_scales[tile_m] = float(right.scale(weight_row, weight_group_index, params));
            if constexpr (HAS_BIAS) {
              weight_offsets[tile_m] = float(right.bias()[right.scale_index(weight_row, weight_group_index, params)]);
            } else {
              weight_offsets[tile_m] = float(right.zero_point(weight_row, weight_group_index, params));
            }
          }
        }

        WeightDecoder<WORDS_PER_LANE> weight_decoder{packed_weight_words};
        WeightFragment weight_fragment{weight_decoder};
        // The correction starts the group product (MMA accumulates onto it), so the fold is one fma per output.
        typename Core::AccumFragment group_product;
        group_product.for_each_lane([&](ushort tile_m, ushort input_index, thread float& product_value) {
          if constexpr (HAS_BIAS) {
            product_value = -activation_sums[2 * input_index + 1];
          } else {
            product_value =
                -fma(weight_offsets[tile_m], activation_sums[2 * input_index], activation_sums[2 * input_index + 1]);
          }
        });
        uzu::matmul::fragment_mma(group_product, weight_fragment, activation_fragment);
        accumulator.for_each_lane(
            [&](ushort tile_m, ushort input_index, thread float& accumulator_value, thread float& group_product_value) {
              if constexpr (HAS_BIAS) {
                accumulator_value =
                    fma(group_product_value,
                        weight_scales[tile_m],
                        fma(weight_offsets[tile_m], activation_sums[2 * input_index], accumulator_value));
              } else {
                accumulator_value = fma(group_product_value, weight_scales[tile_m], accumulator_value);
              }
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

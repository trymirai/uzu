#pragma once

#include <metal_stdlib>

#include "../../../../../common/conversion.h"
#include "../../../../../generated/gemm.h"
#include "../../../../common/fragment.h"
#include "../../../../common/quant_unpack.h"
#include "../../../../common/simdgroup_fragment_ops.h"
#include "../../operands.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

struct DirectSchedule {
  UZU_CONST bool TRANSPOSED_OUTPUT = false;

  template <typename Core>
  static METAL_FUNC typename Core::AccumFragment launch(
      const typename Core::LeftStorage left,
      const typename Core::RightStorage right,
      const constant uzu::matmul::GemmParams* params,
      const thread typename Core::TileContext& tile_context
  ) {
    using RightOperand = typename Core::Right;
    static_assert(
        RightOperand::QUANTIZED && RightOperand::BITS == 8 && RightOperand::GROUP_SIZE == 64,
        "direct schedule requires W8 G64 weights"
    );
    constexpr ushort K_STEPS_PER_BLOCK = 16;
    using Ops = typename Core::FragmentOps;
    constexpr uint K_BLOCK = Core::THREADGROUP_BLOCK_K;
    constexpr uint ACTIVATION_VECTORS_PER_LANE = K_STEPS_PER_BLOCK / 4;

    const uint lane_row = uint(tile_context.lane_position.y);
    const uint lane_column = uint(tile_context.lane_position.x);
    const uint inner_dimension = params->K;
    const uint last_weight_row = params->N - 1;
    const uint weight_sign_mask = right.sign_mask();

    const uint simdgroup_column_base = tile_context.simdgroup_column_base;
    uint weight_rows[Core::TILES_N][2];
    METAL_PRAGMA_UNROLL
    for (uint tile_n = 0; tile_n < Core::TILES_N; ++tile_n) {
      METAL_PRAGMA_UNROLL
      for (uint row_pair = 0; row_pair < 2; ++row_pair) {
        weight_rows[tile_n][row_pair] =
            min(simdgroup_column_base + Ops::FRAGMENT_COLS * tile_n + lane_column + row_pair, last_weight_row);
      }
    }

    typename Core::AccumFragment accumulator;
    accumulator.clear();

    const device bfloat* activation_row =
        left.values + min(tile_context.block_geometry.block_row_start + lane_row, params->M - 1) * inner_dimension +
        K_STEPS_PER_BLOCK * lane_column;
    const auto k_blocks = tile_context.k_blocks;

    uint4 activation_words[ACTIVATION_VECTORS_PER_LANE];
    uint4 weight_words[Core::TILES_N][2];
    float weight_scales[Core::TILES_N][2];
    float dequant_offsets[Core::TILES_N][2];

    auto load_activations = [&](uint k_block, thread uint4(&activation_dst)[ACTIVATION_VECTORS_PER_LANE]) {
      const device uint4* source = reinterpret_cast<const device uint4*>(activation_row + k_block * K_BLOCK);
      METAL_PRAGMA_UNROLL
      for (uint vector = 0; vector < ACTIVATION_VECTORS_PER_LANE; ++vector) {
        activation_dst[vector] = source[vector];
      }
    };
    auto load_weights = [&](uint k_block, thread uint4(&weight_dst)[Core::TILES_N][2]) {
      const uint weight_k_lane = k_block * K_BLOCK + K_STEPS_PER_BLOCK * lane_row;
      METAL_PRAGMA_UNROLL
      for (uint tile_n = 0; tile_n < Core::TILES_N; ++tile_n) {
        METAL_PRAGMA_UNROLL
        for (uint row_pair = 0; row_pair < 2; ++row_pair) {
          weight_dst[tile_n][row_pair] = *reinterpret_cast<const device uint4*>(
              right.row_codes(weight_rows[tile_n][row_pair], params) + weight_k_lane
          );
        }
      }
    };
    auto load_metadata =
        [&](uint k_block, thread float (&scale_dst)[Core::TILES_N][2], thread float (&offset_dst)[Core::TILES_N][2]) {
          const uint weight_group_index = (k_block * K_BLOCK + K_STEPS_PER_BLOCK * lane_row) / RightOperand::GROUP_SIZE;
          METAL_PRAGMA_UNROLL
          for (uint tile_n = 0; tile_n < Core::TILES_N; ++tile_n) {
            METAL_PRAGMA_UNROLL
            for (uint row_pair = 0; row_pair < 2; ++row_pair) {
              const uint row = weight_rows[tile_n][row_pair];
              scale_dst[tile_n][row_pair] = float(right.scale(row, weight_group_index, params));
              offset_dst[tile_n][row_pair] = magic_dequant_bias(scale_dst[tile_n][row_pair]);
            }
          }
        };

    uint k_block = k_blocks.begin + tile_context.simdgroup_k_slice;
    if (k_block < k_blocks.end) {
      load_activations(k_block, activation_words);
      load_weights(k_block, weight_words);
      load_metadata(k_block, weight_scales, dequant_offsets);
    }
    for (; k_block < k_blocks.end; k_block += Core::SIMDGROUPS_PER_K) {
      const uint next_k_block = k_block + Core::SIMDGROUPS_PER_K;
      const uint prefetch_k_block = next_k_block < k_blocks.end ? next_k_block : k_block;
      METAL_PRAGMA_UNROLL
      for (uint tile_n = 0; tile_n < Core::TILES_N; ++tile_n) {
        METAL_PRAGMA_UNROLL
        for (uint row_pair = 0; row_pair < 2; ++row_pair) {
          weight_words[tile_n][row_pair] ^= uint4(weight_sign_mask);
        }
      }

      auto activation_generator = [&](ushort, ushort k_index) -> float2 {
        const uint word0 = reinterpret_cast<const thread uint*>(activation_words)[k_index / 2];
        const uint word1 = reinterpret_cast<const thread uint*>(activation_words)[K_STEPS_PER_BLOCK / 2 + k_index / 2];
        const float2 values0 = uzu::unpack_bfloat(word0);
        const float2 values1 = uzu::unpack_bfloat(word1);
        return k_index % 2 ? float2(values0.y, values1.y) : float2(values0.x, values1.x);
      };
      auto weight_generator = [&](ushort k_index, ushort tile_n) -> float2 {
        const uint word_index = k_index * RightOperand::BITS / 32;
        auto decode = [&](ushort row_pair) {
          return fma(
              magic_code_float<symmetric_zero_point<8>()>(
                  extract_bits(weight_words[tile_n][row_pair][word_index], 8 * (k_index % 4), 8)
              ),
              weight_scales[tile_n][row_pair],
              dequant_offsets[tile_n][row_pair]
          );
        };
        return float2(decode(0), decode(1));
      };
      using ActivationFragment =
          uzu::matmul::GeneratedFragment<float, Core::TILES_M, K_STEPS_PER_BLOCK, Ops, decltype(activation_generator)>;
      using WeightFragment =
          uzu::matmul::GeneratedFragment<float, K_STEPS_PER_BLOCK, Core::TILES_N, Ops, decltype(weight_generator)>;
      ActivationFragment activation_fragment{activation_generator};
      WeightFragment weight_fragment{weight_generator};
      uzu::matmul::fragment_mma(accumulator, activation_fragment, weight_fragment);

      uint4 next_weight_words[Core::TILES_N][2];
      float next_weight_scales[Core::TILES_N][2];
      float next_dequant_offsets[Core::TILES_N][2];
      // Keep next operands separate so their loads can overlap the current MMA.
      load_activations(prefetch_k_block, activation_words);
      load_metadata(prefetch_k_block, next_weight_scales, next_dequant_offsets);
      load_weights(prefetch_k_block, next_weight_words);
      METAL_PRAGMA_UNROLL
      for (uint tile_n = 0; tile_n < Core::TILES_N; ++tile_n) {
        METAL_PRAGMA_UNROLL
        for (uint row_pair = 0; row_pair < 2; ++row_pair) {
          weight_scales[tile_n][row_pair] = next_weight_scales[tile_n][row_pair];
          dequant_offsets[tile_n][row_pair] = next_dequant_offsets[tile_n][row_pair];
          weight_words[tile_n][row_pair] = next_weight_words[tile_n][row_pair];
        }
      }
    }

    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu

#pragma once

#include "../../../common/soft_cap.h"
#include "../../../hadamard_transform/hadamard_transform.h"
#include "arguments.h"
#include "tile.h"

namespace uzu {
namespace gemm {

template <typename Tile, typename AT, typename BT, typename DT, bool FULL_TILE>
struct Epilogue {
  using U = float;

  static METAL_FUNC void store(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile,
      threadgroup U* shared_results
  ) {
    const bool use_hadamard =
        Tile::OUTPUT_ROWS >= METAL_SIMD_SIZE && params.output_transform.contains(GemmDTransform::RHT);
    const bool output_writer = tile.writer && tile.reduction_lane == 0;
    if (!use_hadamard && !output_writer) {
      return;
    }

    if (!use_hadamard) {
      write_results<false>(result, ops, params, tile, shared_results);
      return;
    }
    if (output_writer) {
      write_results<true>(result, ops, params, tile, shared_results);
    }

    if constexpr (Tile::OUTPUT_ROWS >= METAL_SIMD_SIZE) {
      static_assert(Tile::K_SPLIT == 1, "RHT reuses shared results after Reduce");
      threadgroup_barrier(mem_flags::mem_threadgroup);
      constexpr uint OUTPUT_BLOCKS = Tile::OUTPUT_ROWS / METAL_SIMD_SIZE;
      for (uint job = tile.simd_group; job < Tile::INPUT_ROWS * OUTPUT_BLOCKS; job += Tile::NUM_SIMDGROUPS) {
        const uint input_index = job / OUTPUT_BLOCKS;
        const uint output_block = job % OUTPUT_BLOCKS;
        const uint input_row = tile.input_row + input_index;
        const uint global_row = tile.tile_row + output_block * METAL_SIMD_SIZE + tile.simd_lane;
        if ((FULL_TILE || input_row < params.batch_size) && (FULL_TILE || global_row < params.out_vec_size)) {
          DT value = simdgroup_output_random_hadamard_transform(
              static_cast<ushort>(tile.simd_lane),
              static_cast<DT>(
                  shared_results[input_index * Tile::OUTPUT_ROWS + output_block * METAL_SIMD_SIZE + tile.simd_lane]
              ),
              ops.hadamard_factors[global_row]
          );
          if (params.output_transform.contains(GemmDTransform::BIAS)) {
            value += static_cast<DT>(ops.output_bias[global_row]);
          }
          ops.d[input_row * params.out_vec_size + global_row] = value;
        }
      }
    }
  }

private:
  template <bool STAGE_HADAMARD>
  static METAL_FUNC void write_results(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile,
      threadgroup U* shared_results
  ) {
    Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
      constexpr uint I = decltype(input_index)::value;
      const uint input_row = tile.input_row + I;
      if (!(FULL_TILE || input_row < params.batch_size)) {
        return;
      }
      device DT* output = ops.d + input_row * params.out_vec_size + tile.row0;
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        const uint global_row = tile.row0 + R;
        if (!OutputTile<Tile, FULL_TILE>::row_in_range(global_row, params.out_vec_size)) {
          return;
        }
        if (tile.clamped && params.output_transform.contains(GemmDTransform::ACCUMULATE)) {
          return;
        }
        if constexpr (STAGE_HADAMARD) {
          shared_results[I * Tile::OUTPUT_ROWS + tile.local_row + R] =
              transform<false>(result[I][R], output + R, global_row, ops, params);
        } else {
          output[R] = static_cast<DT>(transform<true>(result[I][R], output + R, global_row, ops, params));
        }
      });
    });
  }

  template <bool APPLY_BIAS>
  static METAL_FUNC U transform(
      U value,
      const device DT* output,
      uint global_row,
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params
  ) {
    if (params.output_transform.contains(GemmDTransform::SCALE)) {
      value *= params.ab_scale;
    }
    if (params.output_transform.contains(GemmDTransform::ACCUMULATE)) {
      value += static_cast<U>(*output);
    }
    if constexpr (APPLY_BIAS) {
      if (params.output_transform.contains(GemmDTransform::BIAS)) {
        value += static_cast<U>(ops.output_bias[global_row]);
      }
    }
    if (params.output_transform.contains(GemmDTransform::SOFT_CAP)) {
      value = apply_soft_cap(value, params.soft_cap);
    }
    return value;
  }
};

} // namespace gemm
} // namespace uzu

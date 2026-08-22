#pragma once

#include "tile.h"

namespace uzu {
namespace gemm {

template <typename Tile, bool FULL_TILE>
struct Reduce {
  using U = float;

  static METAL_FUNC void run(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      threadgroup U* shared_results,
      const thread OutputTile<Tile, FULL_TILE>& tile
  ) {
    Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
      constexpr uint I = decltype(input_index)::value;
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        if constexpr (Tile::REDUCTION_LANES == METAL_SIMD_SIZE) {
          result[I][R] = simd_sum(result[I][R]);
        } else {
          U reduced = result[I][R];
          METAL_PRAGMA_UNROLL
          for (uint offset = Tile::REDUCTION_LANES / 2; offset > 0; offset /= 2) {
            reduced += simd_shuffle_xor(reduced, offset);
          }
          result[I][R] = reduced;
        }
      });
    });

    if constexpr (Tile::K_SPLIT > 1) {
      if (tile.reduction_lane == 0) {
        Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
          constexpr uint I = decltype(input_index)::value;
          Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
            constexpr uint R = decltype(output_index)::value;
            const uint row = tile.local_row + R;
            shared_results
                [I * Tile::OUTPUT_ROWS * Tile::K_SPLIT + tile.simd_group * Tile::SIMDGROUP_OUTPUT_ROWS +
                 (row % Tile::SIMDGROUP_OUTPUT_ROWS)] = result[I][R];
          });
        });
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tile.k_slice == 0 && tile.reduction_lane == 0) {
        Tile::for_each_input_row([&](auto input_index) UZU_ALWAYS_INLINE {
          constexpr uint I = decltype(input_index)::value;
          Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
            constexpr uint R = decltype(output_index)::value;
            U sum = 0;
            for (uint slice = 0; slice < Tile::K_SPLIT; slice++) {
              const uint group = tile.row_group * Tile::K_SPLIT + slice;
              const uint row = group * Tile::SIMDGROUP_OUTPUT_ROWS + (tile.local_row % Tile::SIMDGROUP_OUTPUT_ROWS) + R;
              sum += shared_results[I * Tile::OUTPUT_ROWS * Tile::K_SPLIT + row];
            }
            result[I][R] = sum;
          });
        });
      }
    }
  }
};

} // namespace gemm
} // namespace uzu

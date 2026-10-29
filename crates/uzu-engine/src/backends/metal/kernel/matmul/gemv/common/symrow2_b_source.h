#pragma once

#include "arguments.h"
#include "tile.h"

namespace uzu {
namespace gemm {

template <typename Tile, typename AT, typename BT, typename DT, bool FULL_TILE>
struct SymRow2GatherBSource {
  using U = float;

  static METAL_FUNC void run(
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile
  ) {
    static_assert(
        Tile::INPUT_ROWS == 1 && Tile::K_SPLIT == 1,
        "SymRow2 gather reduces each input row in one SIMDgroup"
    );
    static_assert(Tile::OUTPUT_ROWS == 2 * Tile::NUM_SIMDGROUPS, "SymRow2 gather uses 2 candidates per SIMDgroup");
    static_assert(
        Tile::REDUCTION_LANES == 32 && Tile::ROWS_PER_LANE == 2,
        "SymRow2 gather reduces 32 K lanes into 2 outputs"
    );
    // SymRow2 has no scale groups: the group stride carries the number of physical 16-row tiles.
    const uint group_count = params.in_vec_size / 128;
    const uint qv = tile.simd_lane & 3;
    const uint group_slot = tile.simd_lane >> 2;
    const uint input_row = tile.input_row;
    const device AT* x_row = ops.a + input_row * params.in_vec_size;
    const device uint4* weights = reinterpret_cast<const device uint4*>(ops.b);
    uint rows[Tile::ROWS_PER_LANE];
    uint tiles[Tile::ROWS_PER_LANE];
    uint fragment_lanes[Tile::ROWS_PER_LANE];
    uint column_halves[Tile::ROWS_PER_LANE];

    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      const uint candidate = tile.row0 + R;
      rows[R] = ops.gather_indices[input_row * params.out_vec_size + min(candidate, params.out_vec_size - 1)];
      tiles[R] = rows[R] / 16;
      const uint column = rows[R] % 16;
      const uint low_column = column % 8;
      column_halves[R] = column / 8;
      fragment_lanes[R] = (qv & 1) | ((low_column & 3) << 1) | ((qv >> 1) << 3) | ((low_column >> 2) << 4);
    });

    U accumulators[Tile::ROWS_PER_LANE] = {};
    U x_sum = 0.0f;
    for (uint pass = 0; pass < group_count; pass += 8) {
      const uint group = pass + group_slot;
      if (group >= group_count) {
        continue;
      }
      uint4 packed_words[Tile::ROWS_PER_LANE];
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        packed_words[R] = weights[(group * params.scale_group_stride + tiles[R]) * 32 + fragment_lanes[R]];
      });
      U values[32];
      METAL_PRAGMA_UNROLL
      for (uint u = 0; u < 4; u++) {
        METAL_PRAGMA_UNROLL
        for (uint jj = 0; jj < 2; jj++) {
          const device AT* source = x_row + group * 128 + u * 32 + jj * 16 + qv * 4;
          METAL_PRAGMA_UNROLL
          for (uint i = 0; i < 4; i++) {
            const U value = U(source[i]);
            values[u * 8 + jj * 4 + i] = value;
            x_sum += value;
          }
        }
      }

      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        const uint4 packed = packed_words[R];
        U partial = 0.0f;
        METAL_PRAGMA_UNROLL
        for (uint u = 0; u < 4; u++) {
          const uint word = packed[u];
          METAL_PRAGMA_UNROLL
          for (uint jj = 0; jj < 2; jj++) {
            METAL_PRAGMA_UNROLL
            for (uint i = 0; i < 4; i++) {
              const uint code = (word >> (8 * i + 2 * (2 * jj + column_halves[R]))) & 3u;
              partial += values[u * 8 + jj * 4 + i] * U(code);
            }
          }
        }
        accumulators[R] += partial;
      });
    }

    x_sum = simd_sum(x_sum);
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      const U sum = simd_sum(accumulators[R]);
      // Tail candidates repeat the last one: duplicates store equal bytes to the same element.
      const uint candidate = min(tile.row0 + R, params.out_vec_size - 1);
      if (tile.reduction_lane == 0 && input_row < params.batch_size) {
        const U value = (sum - 1.5f * x_sum) * U(ops.scales[rows[R]]);
        ops.d[size_t(input_row) * params.out_vec_size + candidate] = static_cast<DT>(value);
      }
    });
  }
};

} // namespace gemm
} // namespace uzu

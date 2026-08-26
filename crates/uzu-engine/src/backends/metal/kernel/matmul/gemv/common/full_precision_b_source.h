#pragma once

#include "arguments.h"
#include "tile.h"

namespace uzu {
namespace gemm {

template <typename Tile, typename AT, typename BT, typename DT, bool INPUT_ALIGNED, bool FULL_TILE>
struct FullPrecisionBSource {
  using U = float;

  static METAL_FUNC void accumulate(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile
  ) {
    static_assert(Tile::INPUT_ROWS == 1, "full-precision GEMV uses one input row");
    constexpr uint VALUES_PER_THREAD = 4;
    constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * METAL_SIMD_SIZE;
    using W4 = vec<BT, 4>;
    using I4 = vec<AT, 4>;
    const uint k_stride = Tile::K_SPLIT * BLOCK_SIZE;
    const uint k_start = tile.k_slice * BLOCK_SIZE;
    const uint thread_k = tile.reduction_lane * VALUES_PER_THREAD + k_start;
    const uint weights_row_stride = params.in_vec_size;
    const uint input_row = tile.input_row;
    const device AT* input = ops.a + input_row * params.in_vec_size + thread_k;
    thread const device BT* weight_rows[Tile::ROWS_PER_LANE];
    const uint last_row = params.out_vec_size > 0 ? params.out_vec_size - 1 : 0;

    METAL_PRAGMA_UNROLL
    for (uint output_index = 0; output_index < Tile::ROWS_PER_LANE; output_index++) {
      const uint global_row = tile.row0 + output_index;
      const uint lookup_row = FULL_TILE ? global_row : min(global_row, last_row);
      const uint weight_row =
          params.gathered ? ops.gather_indices[input_row * params.out_vec_size + lookup_row] : lookup_row;
      weight_rows[output_index] =
          reinterpret_cast<const device BT*>(ops.b) + weight_row * weights_row_stride + thread_k;
    }

    uint k = k_start;
    for (; k + BLOCK_SIZE <= params.in_vec_size; k += k_stride) {
      const float4 input_values = static_cast<float4>(*reinterpret_cast<const device I4*>(input));
      METAL_PRAGMA_UNROLL
      for (uint output_index = 0; output_index < Tile::ROWS_PER_LANE; output_index++) {
        const uint global_row = tile.row0 + output_index;
        if (FULL_TILE || global_row < params.out_vec_size) {
          result[0][output_index] +=
              dot(static_cast<float4>(*reinterpret_cast<const device W4*>(weight_rows[output_index])), input_values);
        }
        weight_rows[output_index] += k_stride;
      }
      input += k_stride;
    }

    if constexpr (Tile::K_SPLIT == 1 && !INPUT_ALIGNED) {
      const uint thread_offset = tile.reduction_lane * VALUES_PER_THREAD;
      const int remaining =
          k + thread_offset < params.in_vec_size
              ? min(static_cast<int>(params.in_vec_size - k - thread_offset), static_cast<int>(VALUES_PER_THREAD))
              : 0;
      if (remaining > 0) {
        METAL_PRAGMA_UNROLL
        for (uint output_index = 0; output_index < Tile::ROWS_PER_LANE; output_index++) {
          const uint global_row = tile.row0 + output_index;
          if (OutputTile<Tile, FULL_TILE>::row_in_range(global_row, params.out_vec_size)) {
            for (int index = 0; index < remaining; index++) {
              result[0][output_index] +=
                  static_cast<U>(weight_rows[output_index][index]) * static_cast<U>(input[index]);
            }
          }
        }
      }
    }
  }
};

} // namespace gemm
} // namespace uzu

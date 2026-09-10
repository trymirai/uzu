#pragma once

#include "../../common/microfloat.h"
#include "arguments.h"
#include "tile.h"

namespace uzu {
namespace gemm {

template <typename Tile, typename AT, typename BT, typename DT, uint GROUP_SIZE, bool FULL_TILE>
struct MicrofloatBSource {
  using U = float;

  static METAL_FUNC void accumulate(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile
  ) {
    static_assert(Tile::INPUT_ROWS == 1, "microfloat GEMV uses one input row");
    static_assert(Tile::K_SPLIT == 1, "microfloat GEMV does not split K");
    const uint code_row_bytes = params.in_vec_size / 2;
    const uint scale_row_bytes = params.in_vec_size / GROUP_SIZE;
    const float outer_scale = float(ops.outer_scales[0]);
    const uint last_row = params.out_vec_size > 0 ? params.out_vec_size - 1 : 0;

    size_t weight_rows[Tile::ROWS_PER_LANE];
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      const uint output_row = tile.row0 + R;
      const uint lookup_row = FULL_TILE ? output_row : min(output_row, last_row);
      weight_rows[R] =
          params.gathered ? ops.gather_indices[size_t(tile.input_row) * params.out_vec_size + lookup_row] : lookup_row;
    });

    for (uint inner = tile.reduction_lane; inner < params.in_vec_size; inner += Tile::REDUCTION_LANES) {
      const float input_value = float(ops.a[size_t(tile.input_row) * params.in_vec_size + inner]);
      Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
        constexpr uint R = decltype(output_index)::value;
        const uint output_row = tile.row0 + R;
        if (!OutputTile<Tile, FULL_TILE>::row_in_range(output_row, params.out_vec_size)) {
          return;
        }
        const size_t row = weight_rows[R];
        const uint8_t packed = reinterpret_cast<const device uint8_t*>(ops.b)[row * code_row_bytes + inner / 2];
        const uint code = (inner & 1u) == 0 ? packed & 0x0fu : packed >> 4u;
        const uint exponent =
            reinterpret_cast<const device uint8_t*>(ops.scales)[row * scale_row_bytes + inner / GROUP_SIZE];
        const float weight = decode_e2m1(code) * decode_e8m0(exponent) * outer_scale;
        result[0][R] = fma(input_value, weight, result[0][R]);
      });
    }
  }
};

} // namespace gemm
} // namespace uzu

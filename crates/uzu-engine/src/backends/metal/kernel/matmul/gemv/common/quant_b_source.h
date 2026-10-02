#pragma once

#include "arguments.h"
#include "quant_slice.h"
#include "tile.h"

namespace uzu {
namespace gemm {

template <
    typename Tile,
    typename AT,
    typename BT,
    typename DT,
    uint BITS,
    bool FULL_TILE>
struct QuantBSource {
  using U = float;
  using Slice = QuantSlice<Tile, AT, BT, DT, BITS, FULL_TILE>;
  using Metadata = QuantMetadata<Tile, AT, BT, DT, BITS>;

  static METAL_FUNC void accumulate(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile
  ) {
    const uint groups = Metadata::group_count(params, params.group_size);
    const uint row_stride = Slice::row_stride(params);
    const uint values_per_lane = params.group_size / params.group_lanes;
    const uint slices_per_lane = values_per_lane / Slice::SLICE_VALUES;
    const uint groups_per_step = Tile::REDUCTION_LANES / params.group_lanes;
    const uint group_slot = tile.reduction_lane / params.group_lanes;
    const uint group_offset = (tile.reduction_lane % params.group_lanes) * values_per_lane;
    uint weight_row_indices[Tile::ROWS_PER_LANE];
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      const uint output_row = tile.row0 + R;
      weight_row_indices[R] =
          params.gathered ? ops.gather_indices[tile.input_row * params.out_vec_size + output_row] : output_row;
    });

    const device uint8_t* weights = reinterpret_cast<const device uint8_t*>(ops.b);
    const uint batch_remaining = params.batch_size - tile.input_row;
    uint group = group_slot;

    Slice current;
    QuantPosition position = {group, 0};
    while (position.valid(groups)) {
      Metadata metadata;
      metadata.load(position.group, weight_row_indices, ops, params);
      for (position.slice = 0; position.slice < slices_per_lane; position.slice++) {
        current.load_weights(position, weights, weight_row_indices, row_stride, group_offset, params);
        current.accumulate(result, position, ops, params, tile, group_offset, batch_remaining, metadata);
      }
      position.group += groups_per_step;
    }
  }
};

} // namespace gemm
} // namespace uzu

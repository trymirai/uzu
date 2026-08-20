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
    GemmBPrologueKind B_PROLOGUE,
    uint GROUP_SIZE,
    uint BITS,
    bool INPUT_ALIGNED,
    bool PREFETCHED>
struct QuantBSource {
  using U = float;
  using Slice = QuantSlice<Tile, AT, BT, DT, B_PROLOGUE, GROUP_SIZE, BITS, INPUT_ALIGNED>;
  using Metadata = QuantMetadata<Tile, AT, BT, DT, B_PROLOGUE, BITS>;

private:
  UZU_CONST uint VALUES_PER_LANE = GROUP_SIZE / Tile::GROUP_LANES;

public:
  static METAL_FUNC void accumulate(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread Tile& tile
  ) {
    const uint groups = (params.in_vec_size + GROUP_SIZE - 1) / GROUP_SIZE;
    const uint row_stride = params.in_vec_size * BITS / 8;
    const uint group_slot = tile.reduction_lane / Tile::GROUP_LANES;
    const uint group_offset = (tile.reduction_lane % Tile::GROUP_LANES) * VALUES_PER_LANE;
    uint weight_row_indices[Tile::ROWS_PER_LANE];
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      const uint output_row = tile.row0 + R;
      weight_row_indices[R] =
          params.gathered ? ops.gather_indices[tile.input_row * params.out_vec_size + output_row] : output_row;
    });

    const device uint8_t* weights = reinterpret_cast<const device uint8_t*>(ops.b);
    const uint batch_remaining = params.batch_size - tile.input_row;
    typename Slice::Position position = {group_slot, 0};
    if (!position.valid(groups)) {
      return;
    }

    if constexpr (!PREFETCHED) {
      Slice current;
      while (position.valid(groups)) {
        Metadata metadata;
        metadata.load(position.group, groups, weight_row_indices, ops);
        METAL_PRAGMA_UNROLL
        for (uint slice = 0; slice < Slice::SLICES_PER_LANE; slice++) {
          const typename Slice::Position slice_position = {position.group, slice};
          current.load_weights(slice_position, weights, weight_row_indices, row_stride, group_offset);
          current.accumulate(result, slice_position, ops, params, tile, group_offset, batch_remaining, metadata);
        }
        position.group += Tile::GROUPS_PER_STEP;
      }
      return;
    }

    Slice current;
    Slice next;
    Metadata metadata;
    metadata.load(position.group, groups, weight_row_indices, ops);
    current.load_weights(position, weights, weight_row_indices, row_stride, group_offset);
    while (position.valid(groups)) {
      typename Slice::Position next_position = position;
      next_position.advance();
      if (next_position.valid(groups)) {
        next.load_weights(next_position, weights, weight_row_indices, row_stride, group_offset);
      }
      current.accumulate(result, position, ops, params, tile, group_offset, batch_remaining, metadata);
      if (next_position.valid(groups)) {
        current = next;
        if (next_position.group != position.group) {
          metadata.load(next_position.group, groups, weight_row_indices, ops);
        }
      }
      position = next_position;
    }
  }
};

} // namespace gemm
} // namespace uzu

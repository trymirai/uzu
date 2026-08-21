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
    bool FULL_TILE,
    bool PREFETCHED>
struct QuantBSource {
  using U = float;
  using Slice = QuantSlice<Tile, AT, BT, DT, B_PROLOGUE, GROUP_SIZE, BITS, INPUT_ALIGNED, FULL_TILE>;
  using Metadata = QuantMetadata<Tile, AT, BT, DT, B_PROLOGUE, BITS>;

private:
public:
  static METAL_FUNC void accumulate(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile
  ) {
    const uint groups = Metadata::group_count(params, GROUP_SIZE);
    const uint row_stride = Slice::row_stride(params);
    const uint group_slot = tile.reduction_lane / Tile::GROUP_LANES;
    const uint group_offset = (tile.reduction_lane % Tile::GROUP_LANES) * Slice::VALUES_PER_LANE;
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

    if constexpr (!PREFETCHED) {
      Slice current;
      QuantPosition position = {group, 0};
      while (position.valid(groups)) {
        Metadata metadata;
        metadata.load(position.group, groups, weight_row_indices, ops, params);
        for (position.slice = 0; position.slice < Slice::SLICES_PER_LANE; position.slice++) {
          current.load_weights(position, weights, weight_row_indices, row_stride, group_offset);
          current.accumulate(result, position, ops, params, tile, group_offset, batch_remaining, metadata);
        }
        position.group += Tile::GROUPS_PER_STEP;
      }
      return;
    }

    if (group >= groups) {
      return;
    }
    Slice current, next;
    Metadata metadata;
    QuantPosition position = {group, 0};
    current.load_weights(position, weights, weight_row_indices, row_stride, group_offset);
    while (position.valid(groups)) {
      QuantPosition ahead = {position.group, position.slice + 1};
      if (ahead.slice == Slice::SLICES_PER_LANE) {
        ahead.slice = 0;
        ahead.group += Tile::GROUPS_PER_STEP;
      }
      if (ahead.valid(groups)) {
        next.load_weights(ahead, weights, weight_row_indices, row_stride, group_offset);
      }
      // One metadata load per group at slice 0; a second live set costs ~20 regs.
      if (Slice::SLICES_PER_LANE == 1 || position.slice == 0) {
        metadata.load(position.group, groups, weight_row_indices, ops, params);
      }
      current.accumulate(result, position, ops, params, tile, group_offset, batch_remaining, metadata);
      if (ahead.valid(groups)) {
        current = next;
      }
      position = ahead;
    }
  }
};

} // namespace gemm
} // namespace uzu

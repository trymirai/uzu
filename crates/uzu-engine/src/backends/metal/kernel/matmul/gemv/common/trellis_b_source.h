#pragma once

#include "arguments.h"
#include "tile.h"
#include "trellis_slice.h"

namespace uzu {
namespace gemm {

/// QTIP bitshift-trellis weights as a GEMV B source.
///
/// The same shape as `QuantBSource` -- walk K in blocks, load a lane's slice,
/// accumulate it into `result[INPUT_ROWS][ROWS_PER_LANE]` -- with the two
/// differences the format forces:
///
///   * there is no group loop. One scale covers a row, so the metadata is
///     loaded once and folded once, and the K loop accumulates the raw integer
///     dot product (see `TrellisMetadata`).
///   * a lane's K run is CONTIGUOUS rather than group-strided, because the
///     states in it then share one tape load (see `TrellisSlice`).
///
/// The batch stays in registers exactly as it does for the quantized source,
/// and that is what pays for the decode: the trellis codebook is ~9 instructions
/// per weight against a load and a shift for an INT4 code, so it has to be
/// amortised over the batch rather than repeated per batch row.
/// The one step width given a compile-time run: `k = 3` over `V = 4`, the
/// widest the shipped configs use and the one where a compile-time bit offset
/// turns the lane's four states into four INDEPENDENT extractions. Every other
/// `k` takes the runtime walk in `TrellisSlice`.
UZU_CONST uint TRELLIS_STATIC_KV = 12;

template <typename Tile, typename AT, typename BT, typename DT, bool INPUT_ALIGNED, bool FULL_TILE>
struct TrellisBSource {
  using U = float;
  using Metadata = TrellisMetadata<Tile, AT, BT, DT>;

  static_assert(INPUT_ALIGNED, "the trellis GEMV takes only K that is a whole number of blocks");
  static_assert(Tile::K_SPLIT == 1, "the trellis GEMV does not split K");

  static METAL_FUNC void accumulate(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile,
      const constant uzu::matmul::TrellisParams& trellis
  ) {
    if (trellis.k_bits_per_step == TRELLIS_STATIC_KV) {
      run<TRELLIS_STATIC_KV>(result, ops, params, tile, trellis);
    } else {
      run<0>(result, ops, params, tile, trellis);
    }
  }

private:
  template <uint KV>
  static METAL_FUNC void run(
      thread U (&result)[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE],
      const thread GemvOperands<AT, BT, DT>& ops,
      const thread GemvParams& params,
      const thread OutputTile<Tile, FULL_TILE>& tile,
      const constant uzu::matmul::TrellisParams& trellis
  ) {
    using Slice = TrellisSlice<Tile, AT, BT, DT, FULL_TILE, KV>;
    uint weight_row_indices[Tile::ROWS_PER_LANE];
    Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
      constexpr uint R = decltype(output_index)::value;
      weight_row_indices[R] = tile.row0 + R;
    });

    Metadata metadata;
    metadata.load(weight_row_indices, ops, trellis);

    Slice current = Slice::make(
        reinterpret_cast<const device uint*>(ops.b),
        weight_row_indices,
        tile.reduction_lane,
        params.in_vec_size,
        trellis
    );

    const uint blocks = params.in_vec_size / Slice::BLOCK_VALUES;
    const uint batch_remaining = params.batch_size - tile.input_row;
    float partial[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};
    uint column = tile.reduction_lane * Slice::VALUES_PER_LANE;
    for (uint block = 0; block < blocks; block++) {
      current.load_weights();
      current.accumulate(partial, ops, params, tile, column, batch_remaining, trellis);
      current.advance();
      column += Slice::BLOCK_VALUES;
    }

    metadata.fold(result, partial);
  }
};

} // namespace gemm
} // namespace uzu

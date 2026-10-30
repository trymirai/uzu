import definitions;
import matmul.common.matmul;
import matmul.gemv.common.arguments;
import matmul.gemv.common.quant_metadata;
import matmul.gemv.common.quant_slice;
import matmul.gemv.common.tile;

public struct QuantBSource {
  public static void accumulate<A : IModelFloat, B : IModelFloat, D : IModelFloat>(
    inout float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    QuantizedWeights<B> weights,
    GemvOperands<A, B, D> ops,
    GemvParams params,
    GemvTile Tile,
    OutputTile tile,
    bool INPUT_ALIGNED,
    bool FULL_TILE
  ) {
    var current = QuantSlice(Tile, weights.group_size, weights.u4 ? 4u : 8u);
    let groups = QuantMetadata.group_count(params, weights.group_size);
    let row_stride = current.row_stride(params);
    let group_slot = tile.reduction_lane / Tile.GROUP_LANES;
    let group_offset = (tile.reduction_lane % Tile.GROUP_LANES) * current.VALUES_PER_LANE;
    // Rows past the last read the last row, so no load leaves B or the gather indices.
    let last_row = params.out_vec_size > 0 ? params.out_vec_size - 1 : 0;
    uint weight_row_indices[GEMV_MAX_ROWS_PER_LANE];
    [unroll]
    for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
      let output_row = FULL_TILE ? tile.row0 + R : min(tile.row0 + R, last_row);
      weight_row_indices[R] =
        params.gathered ? ops.gather_indices[tile.input_row * params.out_vec_size + output_row] : output_row;
    }

    let batch_remaining = params.batch_size - tile.input_row;
    let group = group_slot;

    QuantPosition position = { group, 0 };
    while (position.valid(groups)) {
      QuantMetadata metadata;
      metadata.load(position.group, weight_row_indices, weights, Tile);
      for (position.slice = 0; position.slice < current.SLICES_PER_LANE; position.slice++) {
        current.load_weights(position, weights, weight_row_indices, row_stride, group_offset, Tile, INPUT_ALIGNED);
        current.accumulate(
          result, position, ops, params, Tile, tile, group_offset, batch_remaining, metadata, weights.method,
          INPUT_ALIGNED, FULL_TILE);
      }
      position.group += Tile.GROUPS_PER_STEP;
    }
  }
}

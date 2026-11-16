import definitions;
import matmul.common.matmul;
import matmul.gemv.common.arguments;
import matmul.gemv.common.quant_metadata;
import matmul.gemv.common.quant_slice;
import matmul.gemv.common.tile;

public struct QuantBSource {
  public static void accumulate<A : IStorageFloat, B : IStorageFloat, D : IStorageFloat>(
    inout float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    QuantizedWeights<B> weights,
    GemvOperands<A, B, D> ops,
    GemmParams params,
    GemvTile Tile,
    OutputTile tile,
    bool INPUT_ALIGNED,
    bool FULL_TILE
  ) {
    var current = QuantSlice(Tile, weights.group_size, weights.u4 ? 4u : 8u);
    let groups = (params.K + weights.group_size - 1) / weights.group_size;
    let row_stride = current.row_stride(params);
    let group_slot = tile.reduction_lane / Tile.GROUP_LANES;
    let group_offset = (tile.reduction_lane % Tile.GROUP_LANES) * current.VALUES_PER_LANE;
    uint weight_row_indices[GEMV_MAX_ROWS_PER_LANE];
    [unroll]
    for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
      let output_row = FULL_TILE ? tile.row0 + R : min(tile.row0 + R, params.N - 1);
      weight_row_indices[R] =
        params.gathered ? params.gathered_row(ops.gather_indices[tile.input_row * params.N + output_row]) : output_row;
    }

    let batch_remaining = params.M - tile.input_row;

    for (uint group = group_slot; group < groups; group += Tile.GROUPS_PER_STEP) {
      QuantMetadata metadata;
      metadata.load(group, weight_row_indices, weights, Tile);
      for (uint slice = 0; slice < current.SLICES_PER_LANE; slice++) {
        let k = group * weights.group_size + group_offset + slice * current.SLICE_VALUES;
        current.load_weights(k, weights, weight_row_indices, row_stride, Tile, INPUT_ALIGNED);
        current.accumulate(
          result, k, ops, params, Tile, tile, batch_remaining, metadata, weights.method, INPUT_ALIGNED, FULL_TILE);
      }
    }
  }
}

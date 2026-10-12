import definitions;
import generated.quantization_method;
import matmul.gemv.common.arguments;
import matmul.gemv.common.quant_metadata;
import matmul.gemv.common.quant_slice;
import matmul.gemv.common.tile;

// Metal gemv/common/quant_b_source.h.

public struct QuantBSource {
  public static void accumulate<A : IModelFloat, B : IModelFloat, D : IModelFloat>(
    inout uint fields,
    inout float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    GemvOperands<A, B, D> ops,
    GemvParams params,
    GemvTile Tile,
    OutputTile tile,
    QuantizationMethod B_PROLOGUE,
    uint GROUP_SIZE,
    uint BITS,
    bool INPUT_ALIGNED,
    bool FULL_TILE
  ) {
    var current = QuantSlice(Tile, GROUP_SIZE, BITS);
    let groups = QuantMetadata.group_count(params, GROUP_SIZE);
    let row_stride = current.row_stride(params);
    let group_slot = tile.reduction_lane / Tile.GROUP_LANES;
    let group_offset = (tile.reduction_lane % Tile.GROUP_LANES) * current.VALUES_PER_LANE;
    // Vulkan only: rows past the last, which Metal reads but never stores when n < ROWS_PER_LANE, read the last row
    // as the full-precision source does, so no load leaves B or the gather indices.
    let last_row = params.out_vec_size > 0 ? params.out_vec_size - 1 : 0;
    uint weight_row_indices[GEMV_MAX_ROWS_PER_LANE];
    [unroll]
    for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
      let output_row = FULL_TILE ? tile.row0 + R : min(tile.row0 + R, last_row);
      weight_row_indices[R] =
        params.gathered ? ops.gather_indices[tile.input_row * params.out_vec_size + output_row] : output_row;
    }

    let weights = gemv_cast<uint8_t, uint>(ops.b);
    let batch_remaining = params.batch_size - tile.input_row;
    let group = group_slot;

    QuantPosition position = { group, 0 };
    while (position.valid(groups)) {
      QuantMetadata metadata;
      metadata.load(fields, position.group, weight_row_indices, ops, params, Tile, B_PROLOGUE, BITS);
      for (position.slice = 0; position.slice < current.SLICES_PER_LANE; position.slice++) {
        current.load_weights(position, weights, weight_row_indices, row_stride, group_offset, Tile, INPUT_ALIGNED);
        current.accumulate(
          fields, result, position, ops, params, Tile, tile, group_offset, batch_remaining, metadata, B_PROLOGUE, INPUT_ALIGNED,
          FULL_TILE);
      }
      position.group += Tile.GROUPS_PER_STEP;
    }
  }
}

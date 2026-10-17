import definitions;
import common.matmul;
import common.storage_address;
import matmul.gemv.common.arguments;
import matmul.gemv.common.tile;

// Metal GEMV store coordinates and epilogue order, with the existing IEEE replay for exceptional operands.
public struct Epilogue {
  public static void store<A : IModelFloat, B : IModelFloat, D : IModelFloat, X : IMatmulOperand, W : IMatmulOperand>(
    float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    GemvOperands<A, B, D> ops,
    GemvParams params,
    GemvTile Tile,
    OutputTile tile,
    uint fields,
    bool FULL_TILE,
    X activations,
    W weights
  ) {
    let output_writer = tile.writer && tile.reduction_lane == 0;
    if (!output_writer) {
      return;
    }
    [unroll]
    for (uint I = 0; I < Tile.INPUT_ROWS; I++) {
      let input_row = tile.input_row + I;
      if (!(FULL_TILE || input_row < params.batch_size)) {
        continue;
      }
      [unroll]
      for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
        let global_row = tile.row0 + R;
        if (!OutputTile.row_in_range(FULL_TILE, global_row, params.out_vec_size)) {
          continue;
        }
        if (global_row < tile.tile_row + tile.local_row) {
          continue;
        }
        let output = element_pointer(ops.d, element_index(input_row, params.out_vec_size, global_row));
        output[0] = D.fromFloat(transform(result[I][R], fields, output, input_row, global_row, ops, params, activations, weights));
      }
    }
  }

  public static float transform<A : IModelFloat, B : IModelFloat, D : IModelFloat, X : IMatmulOperand, W : IMatmulOperand>(
    float value,
    uint fields,
    Ptr<D, Access.ReadWrite> output,
    uint input_row,
    uint global_row,
    GemvOperands<A, B, D> ops,
    GemvParams params,
    X activations,
    W weights
  ) {
    let bias = params.has_bias ? ops.output_bias[global_row].toFloat() : 0.0;
    var dot = value;
    [branch]
    if ((asuint(dot) & 0x7f800000u) == 0x7f800000u || matmul_mode(fields, params.in_vec_size, params.has_scale, params.ab_scale,
      params.has_bias, bias) != MATMUL_FAST) {
      let weight_row = params.gathered ? ops.gather_indices[input_row * params.out_vec_size + global_row] : global_row;
      dot = matmul_replay(activations, input_row, weights, weight_row, params.in_vec_size);
    }
    return matmul_epilogue(
      dot, params.has_scale, params.ab_scale, params.has_bias, bias, params.has_soft_cap, params.soft_cap);
  }
}

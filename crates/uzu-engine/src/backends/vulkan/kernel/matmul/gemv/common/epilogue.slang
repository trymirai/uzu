import definitions;
import matmul.common.matmul;
import matmul.gemv.common.arguments;
import matmul.gemv.common.tile;

public struct Epilogue {
  public static void store<A : IModelFloat, B : IModelFloat, D : IModelFloat>(
    float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    GemvOperands<A, B, D> ops,
    GemvParams params,
    GemvTile Tile,
    OutputTile tile,
    bool FULL_TILE
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
        let bias = params.has_bias ? ops.output_bias[global_row].toFloat() : 0.0;
        let value = matmul_epilogue(
          result[I][R], params.has_scale, params.ab_scale, params.has_bias, bias, params.has_soft_cap, params.soft_cap);
        ops.d[input_row * params.out_vec_size + global_row] = D.fromFloat(value);
      }
    }
  }
}

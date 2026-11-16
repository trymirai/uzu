import definitions;
import matmul.common.matmul;
import matmul.gemv.common.arguments;
import matmul.gemv.common.tile;

public struct Epilogue {
  public static void store<A : IStorageFloat, B : IStorageFloat, D : IStorageFloat>(
    float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    GemvOperands<A, B, D> ops,
    GemmParams params,
    GemvTile Tile,
    OutputTile tile,
    bool FULL_TILE
  ) {
    if (tile.reduction_lane != 0) {
      return;
    }
    [unroll]
    for (uint I = 0; I < Tile.INPUT_ROWS; I++) {
      let input_row = tile.input_row + I;
      if (!(FULL_TILE || input_row < params.M)) {
        continue;
      }
      [unroll]
      for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
        let global_row = tile.row0 + R;
        if (!OutputTile.row_in_range(FULL_TILE, global_row, params.N)) {
          continue;
        }
        if (global_row < tile.tile_row + tile.local_row) {
          continue;
        }
        if (params.gathered && !params.gathers(ops.gather_indices[input_row * params.N + global_row])) {
          continue;
        }
        let bias = params.has_bias ? ops.output_bias[global_row].toFloat() : 0.0;
        ops.d[input_row * params.d_row_stride + global_row] = D.fromFloat(gemm_output(result[I][R], bias, params));
      }
    }
  }
}

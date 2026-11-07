import definitions;
import common.workgroup_reduction;
import matmul.gemv.common.tile;

public struct Reduce {
  public static void run(inout float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE], GemvTile Tile) {
    [unroll]
    for (uint I = 0; I < Tile.INPUT_ROWS; I++) {
      [unroll]
      for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
        result[I][R] = segment_reduce<Sum<float>, GEMV_MAX_SIMDGROUPS * SIMD_SIZE>(result[I][R], SIMD_SIZE);
      }
    }
  }
}

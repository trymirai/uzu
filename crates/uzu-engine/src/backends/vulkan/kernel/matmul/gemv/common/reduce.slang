import definitions;
import common.workgroup_reduction;
import matmul.gemv.common.tile;

// INPUT_ROW_TILE x OUTPUT_ROW_TILE x K_SPLIT entries, at most 32 for the dense kernel's K split.
static const uint GEMV_SHARED_RESULTS = 256;
groupshared float gemv_shared_results[GEMV_SHARED_RESULTS];

public struct Reduce {
  public static void run(
    inout float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    GemvTile Tile,
    OutputTile tile
  ) {
    [unroll]
    for (uint I = 0; I < Tile.INPUT_ROWS; I++) {
      [unroll]
      for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
        result[I][R] =
          segment_reduce<Sum<float>, GEMV_MAX_SIMDGROUPS * SIMD_SIZE>(result[I][R], Tile.REDUCTION_LANES);
      }
    }

    if (Tile.K_SPLIT > 1) {
      if (tile.reduction_lane == 0) {
        [unroll]
        for (uint I = 0; I < Tile.INPUT_ROWS; I++) {
          [unroll]
          for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
            let row = tile.local_row + R;
            gemv_shared_results
              [I * Tile.OUTPUT_ROWS * Tile.K_SPLIT + tile.simd_group * Tile.SIMDGROUP_OUTPUT_ROWS +
               (row % Tile.SIMDGROUP_OUTPUT_ROWS)] = result[I][R];
          }
        }
      }
      GroupMemoryBarrierWithGroupSync();
      if (tile.k_slice == 0 && tile.reduction_lane == 0) {
        [unroll]
        for (uint I = 0; I < Tile.INPUT_ROWS; I++) {
          [unroll]
          for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
            var sum = 0.0;
            for (uint slice = 0; slice < Tile.K_SPLIT; slice++) {
              let group = tile.row_group * Tile.K_SPLIT + slice;
              let row = group * Tile.SIMDGROUP_OUTPUT_ROWS + (tile.local_row % Tile.SIMDGROUP_OUTPUT_ROWS) + R;
              sum += gemv_shared_results[I * Tile.OUTPUT_ROWS * Tile.K_SPLIT + row];
            }
            result[I][R] = sum;
          }
        }
      }
    }
  }
}

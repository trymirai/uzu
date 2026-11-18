import definitions;
import matmul.common.matmul;
import matmul.common.loader;

int2 simdgroup_lane_coordinates(uint simd_lane_id) {
  let quad_index = int(simd_lane_id / 4);
  let lane_row = (quad_index & 4) + int((simd_lane_id / 2) % 4);
  let lane_col = (quad_index & 2) * 2 + int(simd_lane_id % 2) * 2;
  return int2(lane_col, lane_row);
}

public struct ThreadgroupTile<
  A : IStorageFloat,
  B : IStorageFloat,
  let BLOCK_ROWS : int,
  let BLOCK_COLS : int,
  let BLOCK_DEPTH : int,
  let SIMDGROUPS_PER_ROW : int,
  let SIMDGROUPS_PER_COLUMN : int
> {
  public static const int SIMDGROUP_BLOCK_SIZE = 8;
  public static const int TILE_ROWS = BLOCK_ROWS / (SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW);
  public static const int TILE_COLS = BLOCK_COLS / (SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN);

  public static const int A_STRIDE_ROW = BLOCK_DEPTH + 16 / int(sizeof(A));
  public static const int B_STRIDE_COL = BLOCK_DEPTH + 16 / int(sizeof(B));

  public float c[TILE_ROWS * TILE_COLS][2];

  public int simdgroup_row_offset;
  public int simdgroup_col_offset;

  uint a_row_offset;
  uint b_col_offset;

  public __init(ThreadContext thread_context) {
    let tile_row_base = SIMDGROUP_BLOCK_SIZE * (int(thread_context.simdgroup_index) / SIMDGROUPS_PER_COLUMN);
    let tile_col_base = SIMDGROUP_BLOCK_SIZE * (int(thread_context.simdgroup_index) % SIMDGROUPS_PER_COLUMN);

    let simdgroup_coordinates = simdgroup_lane_coordinates(thread_context.simd_lane_id);
    simdgroup_row_offset = simdgroup_coordinates.y;
    simdgroup_col_offset = simdgroup_coordinates.x;

    a_row_offset = uint((tile_row_base + simdgroup_row_offset) * A_STRIDE_ROW);
    b_col_offset = uint((tile_col_base + simdgroup_col_offset) * B_STRIDE_COL);

    simdgroup_row_offset += tile_row_base;
    simdgroup_col_offset += tile_col_base;

    [unroll]
    for (int index = 0; index < TILE_ROWS * TILE_COLS; ++index) {
      c[index][0] = 0.0;
      c[index][1] = 0.0;
    }
  }

  [mutating]
  public void matmul<SA : IThreadgroupStorage<A>, SB : IThreadgroupStorage<B>>() {
    [unroll]
    for (int k_block = 0; k_block < BLOCK_DEPTH; k_block += SIMDGROUP_BLOCK_SIZE) {
      [unroll]
      for (int half = 0; half < SIMDGROUP_BLOCK_SIZE; half += 4) {
        let depth = uint(k_block + half);
        float a[TILE_ROWS][4];
        float b[TILE_COLS][4][2];
        [unroll]
        for (int m = 0; m < TILE_ROWS; ++m) {
          let row = a_row_offset + uint(m * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW * A_STRIDE_ROW) + depth;
          [unroll]
          for (int k = 0; k < 4; ++k) {
            a[m][k] = SA.read(row + uint(k)).toFloat();
          }
        }
        [unroll]
        for (int n = 0; n < TILE_COLS; ++n) {
          let col = b_col_offset + uint(n * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN * B_STRIDE_COL) + depth;
          [unroll]
          for (int k = 0; k < 4; ++k) {
            b[n][k][0] = SB.read(col + uint(k)).toFloat();
            b[n][k][1] = SB.read(col + uint(B_STRIDE_COL + k)).toFloat();
          }
        }
        [unroll]
        for (int m = 0; m < TILE_ROWS; ++m) {
          [unroll]
          for (int n = 0; n < TILE_COLS; ++n) {
            let index = m * TILE_COLS + n;
            [unroll]
            for (int k = 0; k < 4; ++k) {
              c[index][0] = fma(a[m][k], b[n][k][0], c[index][0]);
              c[index][1] = fma(a[m][k], b[n][k][1], c[index][1]);
            }
          }
        }
      }
    }
  }
}

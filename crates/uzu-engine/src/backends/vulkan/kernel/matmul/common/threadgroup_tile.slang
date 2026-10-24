import definitions;
import matmul.common.matmul;
import matmul.common.fragment;
import matmul.common.loader;
import matmul.common.simdgroup_fragment_ops;
import matmul.common.simdgroup_multiply_accumulate;

// A BLOCK_ROWS x BLOCK_COLS FP32 tile of 8 x 8 fragments, A not transposed and B transposed: simdgroup s owns fragment
// row s / SIMDGROUPS_PER_COLUMN + i SIMDGROUPS_PER_ROW and column s % SIMDGROUPS_PER_COLUMN + j SIMDGROUPS_PER_COLUMN.
// A and B are staged as rows of BLOCK_DEPTH padded by 16 bytes.
public struct ThreadgroupTile<
  A : IModelFloat,
  B : IModelFloat,
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

  public Fragment<float, TILE_ROWS, TILE_COLS> c_fragment;

  public int simdgroup_row_offset;
  public int simdgroup_col_offset;

  // This lane's first A row and first B column of the staged tiles, at depth 0.
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

    c_fragment.clear();
  }

  // c += A B over the depth staged in SA and SB: each lane folds c = fma(A[row][k], B[k][col], c) for its elements, k
  // ascending in serpentine order.
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
            let col = (m % 2) != 0 ? (TILE_COLS - 1 - n) : n;
            let index = m * TILE_COLS + col;
            [unroll]
            for (int k = 0; k < 4; ++k) {
              c_fragment.fragment_data[index][0] = fma(a[m][k], b[col][k][0], c_fragment.fragment_data[index][0]);
              c_fragment.fragment_data[index][1] = fma(a[m][k], b[col][k][1], c_fragment.fragment_data[index][1]);
            }
          }
        }
      }
    }
  }

  // The tile from element `d_offset` of `d`.
  public void store_result<D : IModelFloat>(Ptr<D, Access.ReadWrite> d, uint d_offset, uint leading_dimension_d) {
    let destination = d_offset + uint(simdgroup_row_offset) * leading_dimension_d + uint(simdgroup_col_offset);

    fragment_store<D, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, TILE_ROWS, TILE_COLS>(
      c_fragment, d, destination, leading_dimension_d
    );
  }

  public void store_result_safe<D : IModelFloat>(
    Ptr<D, Access.ReadWrite> d,
    uint d_offset,
    uint leading_dimension_d,
    int2 destination_tile_dimensions,
  ) {
    let destination = d_offset + uint(simdgroup_row_offset) * leading_dimension_d + uint(simdgroup_col_offset);
    let dimensions = destination_tile_dimensions - int2(simdgroup_col_offset, simdgroup_row_offset);

    if (dimensions.x <= 0 || dimensions.y <= 0) {
      return;
    }

    fragment_store_safe<D, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, TILE_ROWS, TILE_COLS>(
      c_fragment, d, destination, leading_dimension_d, dimensions
    );
  }
}

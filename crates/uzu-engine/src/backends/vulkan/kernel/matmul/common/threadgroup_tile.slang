import definitions;
import common.matmul;
import common.storage_address;
import matmul.common.defines;
import matmul.common.fragment;
import matmul.common.loader;
import matmul.common.simdgroup_fragment_ops;
import matmul.common.simdgroup_multiply_accumulate;

// Counterpart of Metal matmul/common/threadgroup_tile.h's ThreadgroupTile with an FP32 accumulator, A not transposed and
// B transposed, the only instantiation GEMM uses: simdgroup s owns the 8 x 8 fragments at fragment row
// s / SIMDGROUPS_PER_COLUMN + i SIMDGROUPS_PER_ROW and fragment column s % SIMDGROUPS_PER_COLUMN + j SIMDGROUPS_PER_COLUMN
// of the BLOCK_ROWS x BLOCK_COLS tile, multiplies the staged A and B in 8-deep steps and stores. A and B are staged as
// rows of BLOCK_DEPTH padded by 16 bytes of their storage type; the staging arrays are matmul's type arguments.
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

  public static const int B_STRIDE_INNER = 1;
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

  // c += A B over the staged depth, A staged in SA and B in SB: each lane reads, from the staged tiles, the A row and
  // the two B columns of the elements it owns in each of its fragments and folds c = fma(A[row][k], B[k][col], c) for k
  // ascending in Metal's serpentine order, the sums of Metal's simdgroup_multiply_accumulate on every device.
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
            a[m][k] = staged(SA.read(row + uint(k)));
          }
        }
        [unroll]
        for (int n = 0; n < TILE_COLS; ++n) {
          let col = b_col_offset + uint(n * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN * B_STRIDE_COL) + depth;
          [unroll]
          for (int k = 0; k < 4; ++k) {
            b[n][k][0] = staged(SB.read(col + uint(k)));
            b[n][k][1] = staged(SB.read(col + uint(B_STRIDE_COL + k)));
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

  // A staged value widened from its bits without toFloat's NaN quieting: a NaN or infinity staged anywhere in the tile
  // has exponent field 255, so the finalizer replays every output of the tile from memory; finite values widen alike.
  static float staged<T : IModelFloat>(T value) {
    return T.widened(value.storageBits());
  }

  public void store_result<D : IModelFloat>(Ptr<D, Access.ReadWrite> d, uint leading_dimension_d) {
    let destination =
      element_pointer(d, element_index(uint(simdgroup_row_offset), leading_dimension_d, uint(simdgroup_col_offset)));

    fragment_store<D, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, TILE_ROWS, TILE_COLS>(
      c_fragment, destination, leading_dimension_d
    );
  }

  public void store_result_safe<D : IModelFloat>(
    Ptr<D, Access.ReadWrite> d,
    uint leading_dimension_d,
    int2 destination_tile_dimensions,
  ) {
    let destination =
      element_pointer(d, element_index(uint(simdgroup_row_offset), leading_dimension_d, uint(simdgroup_col_offset)));
    let dimensions = destination_tile_dimensions - int2(simdgroup_col_offset, simdgroup_row_offset);

    if (dimensions.x <= 0 || dimensions.y <= 0) {
      return;
    }

    fragment_store_safe<D, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN, TILE_ROWS, TILE_COLS>(
      c_fragment, destination, leading_dimension_d, dimensions
    );
  }

}

// The multiply of a GEMM block's staged slices: each multiply() adds the A and B slices staged in threadgroup memory to
// its accumulator, and finish() hands the FP32 block to finalization as ThreadgroupTile fragments. Every workgroup
// invocation calls finish once, after the last slice.
public interface ISliceMultiplier<
  A : IModelFloat,
  B : IModelFloat,
  let BLOCK_ROWS : int,
  let BLOCK_COLS : int,
  let BLOCK_DEPTH : int,
  let SIMDGROUPS_PER_ROW : int,
  let SIMDGROUPS_PER_COLUMN : int
> {
  __init(ThreadContext thread_context);
  [mutating] void multiply();
  [mutating] void finish(
    inout ThreadgroupTile<A, B, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN> accumulator
  );
}

// Metal's fragments themselves, from the slices staged in SA and SB: the multiplier for every device.
public struct ScalarMultiplier<
  A : IModelFloat,
  B : IModelFloat,
  SA : IThreadgroupStorage<A>,
  SB : IThreadgroupStorage<B>,
  let BLOCK_ROWS : int,
  let BLOCK_COLS : int,
  let BLOCK_DEPTH : int,
  let SIMDGROUPS_PER_ROW : int,
  let SIMDGROUPS_PER_COLUMN : int
> : ISliceMultiplier<A, B, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN> {
  ThreadgroupTile<A, B, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN> tile;

  public __init(ThreadContext thread_context) {
    tile = ThreadgroupTile<A, B, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN>(
      thread_context
    );
  }

  [mutating]
  public void multiply() {
    tile.matmul<SA, SB>();
  }

  [mutating]
  public void finish(
    inout ThreadgroupTile<A, B, BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN> accumulator
  ) {
    accumulator = tile;
  }
}

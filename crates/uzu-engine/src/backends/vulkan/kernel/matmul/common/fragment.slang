import definitions;
import common.storage_address;

// Counterpart of Metal matmul/common/fragment.h's Fragment over SimdgroupFragmentOps: ROW_FRAGMENTS x COL_FRAGMENTS
// 8 x 8 fragments of T, each lane's two adjacent row elements of fragment (row, col) at row * COL_FRAGMENTS + col.
// Element type as in Metal: A and B fragments hold their storage types, the accumulator FP32.

public struct Fragment<T : IStorageFloat, let ROW_FRAGMENTS : int, let COL_FRAGMENTS : int> {
  public T fragment_data[ROW_FRAGMENTS * COL_FRAGMENTS][2];

  [mutating]
  public void clear() {
    [unroll]
    for (int index = 0; index < ROW_FRAGMENTS * COL_FRAGMENTS; ++index) {
      fragment_data[index][0] = T.fromFloat(0.0);
      fragment_data[index][1] = T.fromFloat(0.0);
    }
  }

  // Metal's load_from of a FragmentSource, `source` at this lane's first element: element (row, column) from row
  // row_stride + column col_stride past it, termwise 64-bit. RAGGED (Metal's bounded source) loads +0, never reading,
  // at or past `bounds` (columns, rows) from that element. A transposed read swaps the strides and bounds, as Metal's
  // ReadTranspose does for SimdgroupFragmentOps.
  [mutating]
  public void load_from<let RAGGED : bool>(Ptr<T, Access.Read> source, uint row_stride, uint col_stride, int2 bounds) {
    [unroll]
    for (int i = 0; i < ROW_FRAGMENTS; ++i) {
      let row = element_pointer(source, element_index(uint(i * 8), row_stride, 0));
      [unroll]
      for (int j = 0; j < COL_FRAGMENTS; ++j) {
        [unroll]
        for (int element = 0; element < 2; element++) {
          var value = T.fromFloat(0.0);
          if (!RAGGED || (i * 8 < bounds.y && j * 8 + element < bounds.x)) {
            value = element_pointer(row, element_index(uint(j * 8 + element), col_stride, 0))[0];
          }
          fragment_data[i * COL_FRAGMENTS + j][element] = value;
        }
      }
    }
  }
}

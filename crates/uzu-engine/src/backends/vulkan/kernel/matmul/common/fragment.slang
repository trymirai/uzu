import definitions;

// ROW_FRAGMENTS x COL_FRAGMENTS 8 x 8 fragments of T, each lane's two adjacent row elements of fragment (row, col) at
// row * COL_FRAGMENTS + col.

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
}

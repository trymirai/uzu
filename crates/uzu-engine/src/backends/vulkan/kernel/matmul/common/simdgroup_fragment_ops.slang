import definitions;
import matmul.common.fragment;

public void fragment_store<U : IStorageFloat, let SIMDGROUP_STRIDE_X : int, let SIMDGROUP_STRIDE_Y : int, let ROWS : int, let COLS : int>(
  Fragment<ROWS, COLS> frag,
  Ptr<U, Access.ReadWrite> destination,
  uint first,
  uint leading_dimension,
) {
  [unroll]
  for (int i = 0; i < ROWS; ++i) {
    [unroll]
    for (int j = 0; j < COLS; ++j) {
      let offset = first + uint(i * 8 * SIMDGROUP_STRIDE_X) * leading_dimension + uint(j * 8 * SIMDGROUP_STRIDE_Y);
      let value = frag.fragment_data[i * COLS + j];
      destination[offset] = U.fromFloat(value[0]);
      destination[offset + 1] = U.fromFloat(value[1]);
    }
  }
}

public void fragment_store_safe<U : IStorageFloat, let SIMDGROUP_STRIDE_X : int, let SIMDGROUP_STRIDE_Y : int, let ROWS : int, let COLS : int>(
  Fragment<ROWS, COLS> frag,
  Ptr<U, Access.ReadWrite> destination,
  uint first,
  uint leading_dimension,
  int2 destination_tile_dimensions,
) {
  [unroll]
  for (int i = 0; i < ROWS; ++i) {
    [unroll]
    for (int j = 0; j < COLS; ++j) {
      let row_offset = i * 8 * SIMDGROUP_STRIDE_X;
      let col_offset = j * 8 * SIMDGROUP_STRIDE_Y;
      let value = frag.fragment_data[i * COLS + j];
      [unroll]
      for (int element = 0; element < 2; element++) {
        if (row_offset < destination_tile_dimensions.y && col_offset + element < destination_tile_dimensions.x) {
          destination[first + uint(row_offset) * leading_dimension + uint(col_offset + element)] = U.fromFloat(value[element]);
        }
      }
    }
  }
}

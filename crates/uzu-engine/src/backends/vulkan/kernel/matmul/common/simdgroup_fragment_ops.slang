import definitions;
import common.matmul;
import matmul.common.fragment;
import matmul.common.loader;
import matmul.common.simdgroup_multiply_accumulate;

// Counterpart of Metal matmul/common/simdgroup_fragment_ops.h's fragment stores to device memory, with SimdgroupMMA's
// stores inlined: a lane's two elements are adjacent in its fragment row. Fragment (i, j) of a simdgroup sits
// SIMDGROUP_STRIDE_X fragments of rows and SIMDGROUP_STRIDE_Y of columns from (i - 1, j) and (i, j - 1), interleaving
// the simdgroups' fragments.

// To `destination`, already offset to this lane's first element, rows leading_dimension apart.
public void fragment_store<U : IModelFloat, let SIMDGROUP_STRIDE_X : int, let SIMDGROUP_STRIDE_Y : int, let ROWS : int, let COLS : int>(
  Fragment<float, ROWS, COLS> frag,
  Ptr<U, Access.ReadWrite> destination,
  uint leading_dimension,
) {
  [unroll]
  for (int i = 0; i < ROWS; ++i) {
    [unroll]
    for (int j = 0; j < COLS; ++j) {
      let offset = uint(i * 8 * SIMDGROUP_STRIDE_X) * leading_dimension + uint(j * 8 * SIMDGROUP_STRIDE_Y);
      let value = frag.fragment_data[i * COLS + j];
      destination[offset] = U.fromFloat(value[0]);
      destination[offset + 1] = U.fromFloat(value[1]);
    }
  }
}

// fragment_store of the elements in rows below destination_tile_dimensions.y and columns below .x, both relative to
// this lane's first element.
public void fragment_store_safe<U : IModelFloat, let SIMDGROUP_STRIDE_X : int, let SIMDGROUP_STRIDE_Y : int, let ROWS : int, let COLS : int>(
  Fragment<float, ROWS, COLS> frag,
  Ptr<U, Access.ReadWrite> destination,
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
          destination[uint(row_offset) * leading_dimension + uint(col_offset + element)] = U.fromFloat(value[element]);
        }
      }
    }
  }
}

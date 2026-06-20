#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

#define ROWS_PER_THREADGROUP 128

// parent:      [B, T] int32, root = -1, parent[i] < i
// path_matrix: [B, T, T] uint8, inclusive ancestor matrix
PUBLIC KERNEL(BuildPathMatrix)(
    const device int32_t* parent,
    device uint8_t* path_matrix,
    constant const uint& batch_size,
    constant const uint& tree_size,
    const uint row_group_id GROUPS((batch_size * tree_size).div_ceil(ROWS_PER_THREADGROUP)),
    const uint row_in_group THREADS(ROWS_PER_THREADGROUP)
) {
  const uint row_id = row_group_id * ROWS_PER_THREADGROUP + row_in_group;
  if (row_id >= batch_size * tree_size) {
    return;
  }

  const uint batch = row_id / tree_size;
  const uint row = row_id - batch * tree_size;
  const uint base = batch * tree_size * tree_size + row * tree_size;

  for (uint col = 0; col < tree_size; ++col) {
    path_matrix[base + col] = 0;
  }

  int cur = int(row);
  for (uint steps = 0; cur >= 0 && steps < tree_size; ++steps) {
    path_matrix[base + uint(cur)] = 1;
    cur = parent[batch * tree_size + uint(cur)];
  }
}

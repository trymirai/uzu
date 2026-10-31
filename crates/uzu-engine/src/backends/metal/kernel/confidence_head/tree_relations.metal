#include <metal_stdlib>
#include "../common/dsl.h"
#include "head.h"
#include "../weaver/weaver_frontier.h"

using namespace metal;


struct TreeRow {
  uint depth;
  uint parent;
  bool present;
};

METAL_FUNC TreeRow tree_row(const device uint* frontier, uint capacity, uint row) {
  if (row == 0) {
    return {0u, 0xffffffffu, true};
  }
  const uint depth = frontier[uint(FrontierIdx::Depth) * capacity + row - 1];
  return {depth, frontier[uint(FrontierIdx::ParentSlot) * capacity + row - 1], depth != 0};
}

METAL_FUNC uint tree_ancestor_row(
    const device uint* slot_ancestors,
    const device uint* slot_rows,
    TreeRow row,
    uint depth,
    uint ancestor_stride
) {
  return slot_rows[depth + 1 == row.depth ? row.parent : slot_ancestors[row.parent * ancestor_stride + depth]];
}

PUBLIC KERNEL(ConfidenceHeadTreeRelations)(
    const device uint* frontier,
    const device uint* slot_ancestors,
    const device uint* slot_rows,
    device uint* relations,
    const constant uint& rows,
    const constant uint& ancestor_stride,
    const uint index AXIS(rows * rows, 256)
) {
  const uint query = index / rows;
  const uint key = index % rows;
  const TreeRow query_row = tree_row(frontier, rows - 1, query);
  const TreeRow key_row = tree_row(frontier, rows - 1, key);
  uint relation = 0;
  if (!query_row.present || !key_row.present) {
    relation = HEAD_RELATION_ABSENT;
  } else if (query == key) {
    relation = 1;
  } else if (key_row.depth < query_row.depth &&
             tree_ancestor_row(slot_ancestors, slot_rows, query_row, key_row.depth, ancestor_stride) == key) {
    relation = key_row.depth + 1 == query_row.depth ? 2 : 4;
  } else if (query_row.depth < key_row.depth &&
             tree_ancestor_row(slot_ancestors, slot_rows, key_row, query_row.depth, ancestor_stride) == query) {
    relation = query_row.depth + 1 == key_row.depth ? 3 : 5;
  } else if (query_row.depth == key_row.depth && query_row.depth != 0 && query_row.parent == key_row.parent) {
    relation = 6;
  }
  relations[index] = relation;
}

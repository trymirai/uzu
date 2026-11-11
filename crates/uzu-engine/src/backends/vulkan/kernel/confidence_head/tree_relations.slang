import definitions;
import confidence_head.head;
import generated.weaver;

// The relation code of each (query, key) pair of tree rows: 1 itself, 2 its parent, 4 another ancestor, 3 its child,
// 5 another descendant, 6 a sibling, 0 unrelated, HEAD_RELATION_ABSENT where either row is absent.

struct TreeRow {
  uint depth;
  uint parent;
  bool present;
}

// Row 0 is the root; row 1 + entry the child of frontier entry `entry`, present where its depth is set.
TreeRow tree_row(Ptr<uint, Access.Read> frontier, uint capacity, uint row) {
  TreeRow result = { 0, HEAD_NONE, true };
  if (row != 0) {
    result.depth = frontier[uint(FrontierIdx.Depth) * capacity + row - 1];
    result.parent = frontier[uint(FrontierIdx.ParentSlot) * capacity + row - 1];
    result.present = result.depth != 0;
  }
  return result;
}

// The tree row of `row`'s ancestor at `depth`.
uint tree_ancestor_row(
  Ptr<uint, Access.Read> slot_ancestors,
  Ptr<uint, Access.Read> slot_rows,
  TreeRow row,
  uint depth,
  uint ancestor_stride,
) {
  return slot_rows[depth + 1 == row.depth ? row.parent : slot_ancestors[row.parent * ancestor_stride + depth]];
}

[[Kernel]] [[Public]]
func ConfidenceHeadTreeRelations(
  Ptr<uint, Access.Read> frontier,
  Ptr<uint, Access.Read> slot_ancestors,
  Ptr<uint, Access.Read> slot_rows,
  Ptr<uint, Access.ReadWrite> relations,
  uint rows,
  uint ancestor_stride,
  [[Axis("rows * rows", "256")]] uint index,
) {
  let query = index / rows;
  let key = index % rows;
  let query_row = tree_row(frontier, rows - 1, query);
  let key_row = tree_row(frontier, rows - 1, key);
  var relation = 0u;
  if (!query_row.present || !key_row.present) {
    relation = HEAD_RELATION_ABSENT;
  } else if (query == key) {
    relation = 1;
  } else if (key_row.depth < query_row.depth
             && tree_ancestor_row(slot_ancestors, slot_rows, query_row, key_row.depth, ancestor_stride) == key) {
    relation = key_row.depth + 1 == query_row.depth ? 2 : 4;
  } else if (query_row.depth < key_row.depth
             && tree_ancestor_row(slot_ancestors, slot_rows, key_row, query_row.depth, ancestor_stride) == query) {
    relation = query_row.depth + 1 == key_row.depth ? 3 : 5;
  } else if (query_row.depth == key_row.depth && query_row.depth != 0 && query_row.parent == key_row.parent) {
    relation = 6;
  }
  relations[index] = relation;
}

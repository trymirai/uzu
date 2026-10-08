import definitions;
import common.attention;
import generated.weaver;

static const uint ANCESTOR_ATTENTION_KEY = 1;
static const uint ANCESTOR_ATTENTION_VALUE = 2;

// One row of one head: the prefix keys, the row's listed ancestor slots of the node cache, then the row's own key, each
// `[.., num_heads, HEAD_DIM]` in planes of keys then values. The row's query and key are rotated at position depth + 1
// and rounded to BF16 as the CPU does before it attends.
struct AncestorKeys : IAttentionKeys {
  Ptr<bf16, Access.Read> prefix_kv;
  Ptr<bf16, Access.ReadWrite> node_kv;
  Ptr<bf16, Access.Read> current_qkv;
  Ptr<float, Access.Read> cosines;
  Ptr<float, Access.Read> sines;
  Ptr<uint, Access.Read> ancestor_indices;
  uint ancestors;
  uint ancestor_count;
  uint prefix_length;
  uint node_capacity;
  uint model_dim;
  uint head_base;
  uint row_base;
  uint table_base;
  uint half_dim;

  // Element j of the row's query or key, rotated and rounded to BF16.
  bf16 rotated(uint component, uint j) {
    let base = row_base + component * model_dim + head_base;
    let first = j < half_dim;
    let paired = current_qkv[base + (first ? j + half_dim : j - half_dim)].toFloat();
    let table = table_base + j;
    return bf16.fromFloat(rope_rotated(current_qkv[base + j].toFloat(), paired, first, cosines[table], sines[table]));
  }

  float query(uint j) {
    return rotated(0, j).toFloat();
  }

  bool used(uint key) {
    return true;
  }

  float key(uint key, uint j) {
    if (key < prefix_length) {
      return prefix_kv[key * model_dim + head_base + j].toFloat();
    }
    if (key - prefix_length < ancestor_count) {
      return node_kv[ancestor_indices[ancestors + key - prefix_length] * model_dim + head_base + j].toFloat();
    }
    return rotated(ANCESTOR_ATTENTION_KEY, j).toFloat();
  }

  float value(uint key, uint j) {
    if (key < prefix_length) {
      return prefix_kv[(prefix_length + key) * model_dim + head_base + j].toFloat();
    }
    if (key - prefix_length < ancestor_count) {
      let slot = ancestor_indices[ancestors + key - prefix_length];
      return node_kv[(node_capacity + slot) * model_dim + head_base + j].toFloat();
    }
    return current_qkv[row_base + ANCESTOR_ATTENTION_VALUE * model_dim + head_base + j].toFloat();
  }
}

// Attention of each row of `current_qkv` `[rows, 3, num_heads, HEAD_DIM]`, one workgroup per row and head, by
// attention_single_query over the prefix `prefix_kv`, the row's first `ancestor_counts[row]` slots of
// `ancestor_indices` `[rows, ancestor_stride]` in `node_kv`, and the row itself, into `output`
// `[rows, num_heads, HEAD_DIM]`; with node_capacity > 0 the row's rotated BF16 key and its value bits are then stored in
// slot `node_indices[row]`. Caller preconditions, which the device does not check: every depth is below max_depth,
// whose rotary rows the tables hold; every count is at most ancestor_stride and every listed slot below node_capacity;
// with node_capacity > 0 every destination slot is below it, and the destinations are distinct and disjoint from every
// slot any row of the dispatch lists, its own row's included.
[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_DIM == 128")]]
func AncestorAttention(
  Ptr<bf16, Access.Read> prefix_kv,
  Ptr<bf16, Access.ReadWrite> node_kv,
  Ptr<bf16, Access.Read> current_qkv,
  Ptr<float, Access.Read> cosines,
  Ptr<float, Access.Read> sines,
  Ptr<uint, Access.Read> node_metadata,
  Ptr<uint, Access.Read> ancestor_indices,
  Ptr<uint, Access.Read> ancestor_counts,
  Ptr<uint, Access.Read> node_indices,
  Ptr<bf16, Access.ReadWrite> output,
  uint rows,
  uint prefix_length,
  uint ancestor_stride,
  uint node_capacity,
  uint max_depth,
  float scale,
  [[Groups("num_heads")]] uint head,
  [[Groups("rows")]] uint row,
  [[Threads("1024")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] uint num_heads,
) {
  static_assert(ATTENTION_THREADS == 1024, "one workgroup of Threads(1024) per row and head");
  let model_dim = num_heads * HEAD_DIM;
  AncestorKeys source;
  source.prefix_kv = prefix_kv;
  source.node_kv = node_kv;
  source.current_qkv = current_qkv;
  source.cosines = cosines;
  source.sines = sines;
  source.ancestor_indices = ancestor_indices;
  source.ancestors = row * ancestor_stride;
  source.ancestor_count = ancestor_counts[row];
  source.prefix_length = prefix_length;
  source.node_capacity = node_capacity;
  source.model_dim = model_dim;
  source.head_base = head * HEAD_DIM;
  source.row_base = row * 3 * model_dim;
  source.table_base = (node_metadata[uint(MetadataIdx.Depth) * rows + row] + 1) * HEAD_DIM;
  source.half_dim = HEAD_DIM / 2;
  let sequence_length = prefix_length + source.ancestor_count + 1;
  attention_single_query(
    source, HEAD_DIM, sequence_length, scale, false, 0.0, output, (row * num_heads + head) * HEAD_DIM, tid
  );
  // Every invocation has read its keys and values: the row's key and value enter its slot.
  if (node_capacity > 0 && tid < HEAD_DIM) {
    let slot = node_indices[row] * model_dim + source.head_base + tid;
    let value = source.row_base + ANCESTOR_ATTENTION_VALUE * model_dim + source.head_base + tid;
    node_kv[slot] = source.rotated(ANCESTOR_ATTENTION_KEY, tid);
    node_kv[node_capacity * model_dim + slot] = current_qkv[value];
  }
}

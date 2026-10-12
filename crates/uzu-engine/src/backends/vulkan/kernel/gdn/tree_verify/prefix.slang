import definitions;
import common.activation;
import common.storage_address;
import generated.trie;

// The CPU BuildTreePrefix. prefix[b, row, head] is the FP32 sum from +0 over columns col ascending whose trie
// interval [trie_start, trie_end] holds row (col is an ancestor of row or row itself) of log_decay[b, col, head], each
// sum an IEEE FP32 operation (ordered_sum) in the CPU's order rather than Metal's parallel reduction. One invocation
// owns each (row, head) and writes it once per batch; the batch is a loop inside the shader. All rows use the same trie
// node and log_decay address for each column, loading the latter only for an ancestor. Element offsets are 64-bit words formed one term at a
// time. Workgroups stride over row blocks and heads by the actual NumWorkgroups, which the host clamps to the device
// limits, advancing only while more than one step remains.

static const uint BUILD_TREE_PREFIX_THREADS = 64;

[[Kernel]] [[Public]]
func BuildTreePrefix(
  Ptr<TrieNode, Access.Read> trie,
  Ptr<float, Access.Read> log_decay,
  Ptr<float, Access.ReadWrite> prefix,
  uint batch_size,
  uint tree_size,
  uint value_heads,
  [[Groups("if batch_size == 0 || tree_size == 0 || value_heads == 0 { 0 } else { tree_size.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint row_group,
  [[Groups("if batch_size == 0 || tree_size == 0 { 0 } else { value_heads.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1]) }")]] uint head_group,
  [[Threads("64")]] uint thread_idx,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let blocks = tree_size / BUILD_TREE_PREFIX_THREADS + (tree_size % BUILD_TREE_PREFIX_THREADS != 0 ? 1u : 0u);
  var head = head_group;
  while (head < value_heads) {
    var block = row_group;
    while (block < blocks) {
      if (thread_idx < tree_size - block * BUILD_TREE_PREFIX_THREADS) {
        let row = block * BUILD_TREE_PREFIX_THREADS + thread_idx;
        for (uint b = 0; b < batch_size; b++) {
          var sum = 0.0;
          for (uint col = 0; col < tree_size; col++) {
            let node = element_pointer(trie, element_index(b, tree_size, col))[0];
            if (row >= node.trie_start && row <= node.trie_end) {
              let element = add_address(byte_offset(element_index(b, tree_size, col), value_heads), uint2(head, 0));
              sum = ordered_sum(sum, element_pointer(log_decay, element)[0]);
            }
          }
          let element = add_address(byte_offset(element_index(b, tree_size, row), value_heads), uint2(head, 0));
          element_pointer(prefix, element)[0] = sum;
        }
      }
      if (blocks - block <= groups.x) {
        break;
      }
      block += groups.x;
    }
    if (value_heads - head <= groups.y) {
      break;
    }
    head += groups.y;
  }
}

import definitions;
import common.activation;
import common.storage_address;
import generated.weaver;

// The CPU's explicit path_logprob_bits contract: propagate signaling child/parent before quiet child/parent.
// Return integer bits so native floating-point arithmetic cannot replace an observable NaN payload or sign.
uint weaver_path_logprob_bits(uint parent, uint child) {
  let parent_nan = (parent & 0x7fffffffu) > 0x7f800000u;
  let child_nan = (child & 0x7fffffffu) > 0x7f800000u;
  if (child_nan && (child & 0x00400000u) == 0) { return child | 0x00400000u; }
  if (parent_nan && (parent & 0x00400000u) == 0) { return parent | 0x00400000u; }
  if (child_nan) { return child | 0x00400000u; }
  if (parent_nan) { return parent | 0x00400000u; }
  if ((parent & 0x7fffffffu) == 0x7f800000u && child == (parent ^ 0x80000000u)) { return 0x7fc00000u; }
  return asuint(exact_sum(asfloat(parent), asfloat(child), false));
}

// Disjoint buffers with the CPU's full declared shapes; parent depth + 1 must be defined by the CPU contract.
// One invocation owns a destination slot and takes its last valid matching row; every output word has one writer.
[[Kernel]] [[Public]]
func WeaverFrontierInsertChildren(
  Ptr<uint, Access.Read> packed_tree,
  Ptr<uint, Access.Read> node_metadata,
  Ptr<uint, Access.Read> node_valid,
  Ptr<uint, Access.Read> child_ids,
  Ptr<float, Access.Read> child_logprobs,
  Ptr<float, Access.Read> child_prune_logprobs,
  Ptr<uint, Access.ReadWrite> frontier,
  uint frontier_capacity,
  uint tree_slot_count,
  uint node_count,
  uint expand_width,
  [[Groups("if frontier_capacity == 0 || tree_slot_count == 0 || node_count == 0 || expand_width == 0 { 0 } else { ((frontier_capacity - 1) / 256 + 1).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint group,
  [[Threads("256")]] uint tid,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  var slot = group * 256 + tid;
  while (slot < frontier_capacity) {
    let parent = slot / expand_width;
    let column = slot % expand_width;
    if (parent < tree_slot_count) {
      var row = node_count;
      while (row != 0) {
        row--;
        if (node_valid[row] == 0) { continue; }
        let source_parent = element_pointer(node_metadata, element_index(uint(MetadataIdx.TreeSlot), node_count, row))[0];
        if (source_parent != parent) { continue; }
        let source = element_index(row, expand_width, column);
        let parent_bits = element_pointer(packed_tree, element_index(uint(TreeIdx.PathLogprobBits), tree_slot_count, parent))[0];
        let sum_bits = weaver_path_logprob_bits(parent_bits, asuint(element_pointer(child_logprobs, source)[0]));
        let depth = element_pointer(packed_tree, element_index(uint(TreeIdx.Depth), tree_slot_count, parent))[0] + 1;
        let score_key = (sum_bits & 0x80000000u) != 0 ? ~sum_bits : sum_bits ^ 0x80000000u;
        element_pointer(frontier, element_index(uint(FrontierIdx.TokenId), frontier_capacity, slot))[0] = element_pointer(child_ids, source)[0];
        element_pointer(frontier, element_index(uint(FrontierIdx.ParentSlot), frontier_capacity, slot))[0] = parent;
        element_pointer(frontier, element_index(uint(FrontierIdx.Depth), frontier_capacity, slot))[0] = depth;
        element_pointer(frontier, element_index(uint(FrontierIdx.PathLogprobBits), frontier_capacity, slot))[0] = sum_bits;
        element_pointer(frontier, element_index(uint(FrontierIdx.EdgeLogprobBits), frontier_capacity, slot))[0] = asuint(element_pointer(child_prune_logprobs, source)[0]);
        element_pointer(frontier, element_index(uint(FrontierIdx.PathScoreKey), frontier_capacity, slot))[0] = score_key;
        element_pointer(frontier, element_index(uint(FrontierIdx.Active), frontier_capacity, slot))[0] = 1;
        break;
      }
    }
    if (groups.x > (frontier_capacity - 1 - slot) / 256) { break; }
    slot += groups.x * 256;
  }
}

import definitions;
import common.storage_address;
import generated.weaver;

groupshared uint4 weaver_select_reduce[FRONTIER_SELECT_THREADS];
groupshared uint weaver_select_winners[FRONTIER_MAX_WIDTH];

uint4 weaver_select_better(uint4 a, uint4 b) {
  return b.x > a.x || (b.x == a.x && (b.y < a.y ||
      (b.y == a.y && (b.z < a.z || (b.z == a.z && b.w < a.w))))) ? b : a;
}

uint4 weaver_select_wave(uint4 value) {
  uint4 best;
  best.x = WaveActiveMax(value.x);
  best.y = WaveActiveMin(value.x == best.x ? value.y : FRONTIER_NO_WINNER);
  best.z = WaveActiveMin(all(value.xy == best.xy) ? value.z : FRONTIER_NO_WINNER);
  best.w = WaveActiveMin(all(value.xyz == best.xyz) ? value.w : FRONTIER_NO_WINNER);
  return best;
}

// Complete, distinct CPU-shaped buffers. F32 candidate logits are copied as words, preserving every payload bit.
[[Kernel]] [[Public]]
func WeaverFrontierSelect(
  Ptr<uint, Access.ReadWrite> frontier,
  Ptr<uint, Access.ReadWrite> packed_tree,
  Ptr<uint, Access.ReadWrite> slot_ancestors,
  Ptr<uint, Access.ReadWrite> node_token_ids,
  Ptr<uint, Access.ReadWrite> node_metadata,
  Ptr<uint, Access.ReadWrite> node_ancestor_indices,
  Ptr<uint, Access.ReadWrite> node_valid,
  Ptr<uint, Access.Read> candidate_pool_ids,
  Ptr<uint, Access.Read> candidate_pool_logits,
  Ptr<uint, Access.ReadWrite> node_candidate_ids,
  Ptr<uint, Access.ReadWrite> node_candidate_logits,
  uint frontier_capacity,
  uint tree_slot_count,
  uint node_count,
  uint batch_start_slot,
  uint ancestor_stride,
  uint max_depth,
  uint lookahead_count,
  uint candidate_depth_count,
  uint candidates_per_depth,
  [[Groups("if frontier_capacity == 0 || frontier_capacity > crate::backends::common::gpu_types::weaver::FRONTIER_MAX_SLOTS || node_count == 0 || node_count > crate::backends::common::gpu_types::weaver::FRONTIER_MAX_WIDTH || ancestor_stride == 0 || max_depth == 0 || tree_slot_count == 0 || batch_start_slot + node_count > tree_slot_count || candidate_depth_count == 0 || candidates_per_depth == 0 { 0 } else { 1 }")]] uint group,
  [[Threads("256")]] uint tid,
) {
  static_assert(FRONTIER_SELECT_THREADS == 256, "selection workgroup must match the binding");
  static_assert(FRONTIER_ENTRIES_PER_THREAD * FRONTIER_SELECT_THREADS == FRONTIER_MAX_SLOTS, "selection entries must cover the frontier");
  static_assert(FRONTIER_MAX_WIDTH <= FRONTIER_SELECT_THREADS, "one invocation must own each output node");
  let neutral = uint4(0, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER);
  uint4 entries[FRONTIER_ENTRIES_PER_THREAD];
  for (uint entry = 0; entry < FRONTIER_ENTRIES_PER_THREAD; entry++) {
    let slot = tid + entry * FRONTIER_SELECT_THREADS;
    entries[entry] = neutral;
    if (slot >= frontier_capacity) { continue; }
    if (frontier[uint(FrontierIdx.Active) * frontier_capacity + slot] == 0) { continue; }
    let depth = frontier[uint(FrontierIdx.Depth) * frontier_capacity + slot];
    let score = frontier[uint(FrontierIdx.PathScoreKey) * frontier_capacity + slot];
    let parent = frontier[uint(FrontierIdx.ParentSlot) * frontier_capacity + slot];
    let token = frontier[uint(FrontierIdx.TokenId) * frontier_capacity + slot];
    let next = uint4(((depth < lookahead_count ? 1u : 0u) << 31) | (score >> 1), parent, token, slot);
    // The CPU compares only the first three keys against neutral; an exact neutral triple never wins.
    if (any(next.xyz != neutral.xyz)) { entries[entry] = next; }
  }
  for (uint node = 0; node < node_count; node++) {
    var local = neutral;
    for (uint entry = 0; entry < FRONTIER_ENTRIES_PER_THREAD; entry++) {
      local = weaver_select_better(local, entries[entry]);
    }
    let partial = weaver_select_wave(local);
    let lane = WavePrefixSum(1u);
    let lanes = WaveActiveSum(1u);
    if (lane == 0) { weaver_select_reduce[subgroup_id()] = partial; }
    GroupMemoryBarrierWithGroupSync();
    if (subgroup_id() == 0) {
      var total = neutral;
      for (uint subgroup = lane; subgroup < subgroup_count(); subgroup += lanes) {
        total = weaver_select_better(total, weaver_select_reduce[subgroup]);
      }
      total = weaver_select_wave(total);
      if (lane == 0) { weaver_select_winners[node] = total.w; }
    }
    GroupMemoryBarrierWithGroupSync();
    let winner = weaver_select_winners[node];
    for (uint entry = 0; entry < FRONTIER_ENTRIES_PER_THREAD; entry++) {
      if (entries[entry].w == winner) { entries[entry] = neutral; }
    }
  }

  if (tid < node_count) {
    let slot = weaver_select_winners[tid];
    let real = slot != FRONTIER_NO_WINNER;
    let source = real ? slot : 0;
    let tree_slot = batch_start_slot + tid;
    let token = real ? frontier[uint(FrontierIdx.TokenId) * frontier_capacity + source] : 0;
    let parent = real ? frontier[uint(FrontierIdx.ParentSlot) * frontier_capacity + source] : FRONTIER_NO_WINNER;
    let depth = real ? frontier[uint(FrontierIdx.Depth) * frontier_capacity + source] : 0;
    let path = real ? frontier[uint(FrontierIdx.PathLogprobBits) * frontier_capacity + source] : 0;
    let edge = real ? frontier[uint(FrontierIdx.EdgeLogprobBits) * frontier_capacity + source] : 0;
    element_pointer(packed_tree, element_index(uint(TreeIdx.TokenId), tree_slot_count, tree_slot))[0] = token;
    element_pointer(packed_tree, element_index(uint(TreeIdx.ParentSlot), tree_slot_count, tree_slot))[0] = parent;
    element_pointer(packed_tree, element_index(uint(TreeIdx.Depth), tree_slot_count, tree_slot))[0] = depth;
    element_pointer(packed_tree, element_index(uint(TreeIdx.PathLogprobBits), tree_slot_count, tree_slot))[0] = path;
    element_pointer(packed_tree, element_index(uint(TreeIdx.EdgeLogprobBits), tree_slot_count, tree_slot))[0] = edge;
    element_pointer(packed_tree, element_index(uint(TreeIdx.Valid), tree_slot_count, tree_slot))[0] = real ? 1 : 0;
    if (real) { frontier[uint(FrontierIdx.Active) * frontier_capacity + slot] = 0; }
    node_token_ids[tid] = token;
    node_metadata[uint(MetadataIdx.Depth) * node_count + tid] = depth < lookahead_count ? depth : PADDING_DEPTH;
    node_metadata[uint(MetadataIdx.AncestorCount) * node_count + tid] = depth;
    node_metadata[uint(MetadataIdx.TreeSlot) * node_count + tid] = tree_slot;
    node_valid[tid] = real && depth < lookahead_count ? 1 : 0;
  }

  // Each column's owner visits nodes in CPU order, including earlier, later and self parent/output overlaps.
  // All storage dependencies stay in that invocation; the selection barriers protect only shared scratch.
  var column = tid;
  while (column < ancestor_stride) {
    for (uint node = 0; node < node_count; node++) {
      let slot = weaver_select_winners[node];
      let real = slot != FRONTIER_NO_WINNER;
      let source = real ? slot : 0;
      let depth = real ? frontier[uint(FrontierIdx.Depth) * frontier_capacity + source] : 0;
      let parent = real ? frontier[uint(FrontierIdx.ParentSlot) * frontier_capacity + source] : FRONTIER_NO_WINNER;
      let parent_slot = real && parent < tree_slot_count ? parent : 0;
      uint ancestor = 0;
      if (real && column + 1 < depth) {
        ancestor = element_pointer(slot_ancestors, element_index(parent_slot, ancestor_stride, column))[0];
      } else if (real && column + 1 == depth) {
        ancestor = parent_slot;
      }
      element_pointer(slot_ancestors, element_index(batch_start_slot + node, ancestor_stride, column))[0] = ancestor;
      element_pointer(node_ancestor_indices, element_index(node, ancestor_stride, column))[0] = ancestor;
    }
    if (FRONTIER_SELECT_THREADS > ancestor_stride - 1 - column) { break; }
    column += FRONTIER_SELECT_THREADS;
  }
  for (uint node = 0; node < node_count; node++) {
    let slot = weaver_select_winners[node];
    let depth = slot != FRONTIER_NO_WINNER ? frontier[uint(FrontierIdx.Depth) * frontier_capacity + slot] : 0;
    if (depth >= candidate_depth_count) { continue; }
    var candidate = tid;
    while (candidate < candidates_per_depth) {
      let source = element_index(depth, candidates_per_depth, candidate);
      let destination = element_index(node, candidates_per_depth, candidate);
      element_pointer(node_candidate_ids, destination)[0] = element_pointer(candidate_pool_ids, source)[0];
      element_pointer(node_candidate_logits, destination)[0] = element_pointer(candidate_pool_logits, source)[0];
      if (FRONTIER_SELECT_THREADS > candidates_per_depth - 1 - candidate) { break; }
      candidate += FRONTIER_SELECT_THREADS;
    }
  }
}

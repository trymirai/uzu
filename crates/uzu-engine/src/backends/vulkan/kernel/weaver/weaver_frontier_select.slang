import definitions;
import common.ranking;
import common.workgroup_reduction;
import generated.weaver;

groupshared uint weaver_select_winners[FRONTIER_MAX_WIDTH];

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
      local = WeaverRank.combine(local, entries[entry]);
    }
    let winner = workgroup_reduce<WeaverRank, FRONTIER_SELECT_THREADS>(local).w;
    if (tid == 0) { weaver_select_winners[node] = winner; }
    for (uint entry = 0; entry < FRONTIER_ENTRIES_PER_THREAD; entry++) {
      if (entries[entry].w == winner) { entries[entry] = neutral; }
    }
  }
  GroupMemoryBarrierWithGroupSync();

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
    packed_tree[uint(TreeIdx.TokenId) * tree_slot_count + tree_slot] = token;
    packed_tree[uint(TreeIdx.ParentSlot) * tree_slot_count + tree_slot] = parent;
    packed_tree[uint(TreeIdx.Depth) * tree_slot_count + tree_slot] = depth;
    packed_tree[uint(TreeIdx.PathLogprobBits) * tree_slot_count + tree_slot] = path;
    packed_tree[uint(TreeIdx.EdgeLogprobBits) * tree_slot_count + tree_slot] = edge;
    packed_tree[uint(TreeIdx.Valid) * tree_slot_count + tree_slot] = real ? 1 : 0;
    if (real) { frontier[uint(FrontierIdx.Active) * frontier_capacity + slot] = 0; }
    node_token_ids[tid] = token;
    node_metadata[uint(MetadataIdx.Depth) * node_count + tid] = depth < lookahead_count ? depth : PADDING_DEPTH;
    node_metadata[uint(MetadataIdx.AncestorCount) * node_count + tid] = depth;
    node_metadata[uint(MetadataIdx.TreeSlot) * node_count + tid] = tree_slot;
    node_valid[tid] = real && depth < lookahead_count ? 1 : 0;
  }

  // Each column's owner visits nodes in CPU order, including earlier, later and self parent/output overlaps.
  // All storage dependencies stay in that invocation; the selection barriers protect only shared scratch.
  for (uint column = tid; column < ancestor_stride; column += FRONTIER_SELECT_THREADS) {
    for (uint node = 0; node < node_count; node++) {
      let slot = weaver_select_winners[node];
      let real = slot != FRONTIER_NO_WINNER;
      let source = real ? slot : 0;
      let depth = real ? frontier[uint(FrontierIdx.Depth) * frontier_capacity + source] : 0;
      let parent = real ? frontier[uint(FrontierIdx.ParentSlot) * frontier_capacity + source] : FRONTIER_NO_WINNER;
      let parent_slot = real && parent < tree_slot_count ? parent : 0;
      uint ancestor = 0;
      if (real && column + 1 < depth) {
        ancestor = slot_ancestors[parent_slot * ancestor_stride + column];
      } else if (real && column + 1 == depth) {
        ancestor = parent_slot;
      }
      slot_ancestors[(batch_start_slot + node) * ancestor_stride + column] = ancestor;
      node_ancestor_indices[node * ancestor_stride + column] = ancestor;
    }
  }
  for (uint node = 0; node < node_count; node++) {
    let slot = weaver_select_winners[node];
    let depth = slot != FRONTIER_NO_WINNER ? frontier[uint(FrontierIdx.Depth) * frontier_capacity + slot] : 0;
    if (depth >= candidate_depth_count) { continue; }
    let source = depth * candidates_per_depth;
    let destination = node * candidates_per_depth;
    for (uint candidate = tid; candidate < candidates_per_depth; candidate += FRONTIER_SELECT_THREADS) {
      node_candidate_ids[destination + candidate] = candidate_pool_ids[source + candidate];
      node_candidate_logits[destination + candidate] = candidate_pool_logits[source + candidate];
    }
  }
}

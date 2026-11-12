import definitions;
import common.sampling;
import common.ranking;
import common.workgroup_reduction;
import generated.weaver;

groupshared uint weaver_select_winners[FRONTIER_MAX_WIDTH];

void weaver_children(
  Ptr<bf16, Access.Read> residual_logits,
  Ptr<uint, Access.Read> depth_seeds,
  Ptr<uint, Access.ReadWrite> frontier,
  Ptr<uint, Access.ReadWrite> packed_tree,
  Ptr<uint, Access.ReadWrite> node_metadata,
  Ptr<uint, Access.Read> candidate_pool_ids,
  Ptr<uint, Access.Read> candidate_pool_logits,
  Ptr<uint, Access.ReadWrite> node_candidate_ids,
  Ptr<uint, Access.ReadWrite> node_candidate_logits,
  bool from_pool,
  uint rows,
  uint row,
  uint lid,
  uint candidates,
  uint expand_width,
  uint vocab_size,
  uint frontier_capacity,
  uint tree_slot_count,
  float prune_noise_scale,
  bool has_prune_noise,
) {
  let tree_slot = node_metadata[uint(MetadataIdx.TreeSlot) * rows + row];
  let frontier_base = tree_slot * expand_width;
  let base = row * candidates;
  let depth = node_metadata[uint(MetadataIdx.Depth) * rows + row];
  let key = uint2(depth_seeds[depth * 2], depth_seeds[depth * 2 + 1]);
  let neutral = uint4(0, 0xffffffffu, 0xffffffffu, 0xffffffffu);
  float logits[2];
  float prune_logits[2];
  uint4 entries[2];
  for (uint cache = 0; cache < 2; cache++) {
    let index = lid + cache * TOP_CHILDREN_THREADS;
    logits[cache] = NEGATIVE_INFINITY;
    prune_logits[cache] = NEGATIVE_INFINITY;
    entries[cache] = neutral;
    if (index >= candidates) { continue; }
    let token = from_pool ? candidate_pool_ids[base + index] : node_candidate_ids[base + index];
    let logit = from_pool ? candidate_pool_logits[base + index] : node_candidate_logits[base + index];
    let sample_slot = sampling_index(token, vocab_size);
    let word = sampling_words(key, sample_slot.x)[sample_slot.y];
    let residual_logit = asfloat(logit) + residual_logits[base + index].toFloat();
    logits[cache] = residual_logit;
    entries[cache] = uint4(
      float_order_key(asuint(residual_logit + sampling_noise(float(word) * (1.0 / 4294967296.0)))), token, index, index
    );
    if (has_prune_noise) {
      prune_logits[cache] = residual_logit + prune_noise_scale * sampling_noise(sampling_uniform(word));
    }
  }
  let maximum = workgroup_reduce<Max<float>, TOP_CHILDREN_THREADS>(max(logits[0], logits[1]));
  float local_sum = 0.0;
  for (uint cache = 0; cache < 2; cache++) {
    if (lid + cache * TOP_CHILDREN_THREADS < candidates) {
      local_sum += exp(logits[cache] - maximum);
    }
  }
  let log_sum = log(workgroup_reduce<Sum<float>, TOP_CHILDREN_THREADS>(local_sum)) + maximum;
  float prune_log_sum = 0.0;
  if (has_prune_noise) {
    let prune_maximum =
      workgroup_reduce<Max<float>, TOP_CHILDREN_THREADS>(max(prune_logits[0], prune_logits[1]));
    float prune_local_sum = 0.0;
    for (uint cache = 0; cache < 2; cache++) {
      if (lid + cache * TOP_CHILDREN_THREADS < candidates) {
        prune_local_sum += exp(prune_logits[cache] - prune_maximum);
      }
    }
    prune_log_sum =
      log(workgroup_reduce<Sum<float>, TOP_CHILDREN_THREADS>(prune_local_sum)) + prune_maximum;
  }
  let parent_bits = packed_tree[uint(TreeIdx.PathLogprobBits) * tree_slot_count + tree_slot];
  let child_depth = packed_tree[uint(TreeIdx.Depth) * tree_slot_count + tree_slot] + 1;
  for (uint child = 0; child < expand_width; child++) {
    let winner = workgroup_reduce<WeaverRank, TOP_CHILDREN_THREADS>(WeaverRank.combine(entries[0], entries[1]));
    let slot = frontier_base + child;
    for (uint cache = 0; cache < 2; cache++) {
      if (lid + cache * TOP_CHILDREN_THREADS != winner.z) { continue; }
      if (slot < frontier_capacity) {
        let model_logprob = logits[cache] - log_sum;
        let path_bits = asuint(asfloat(parent_bits) + model_logprob);
        let edge_logprob = has_prune_noise ? prune_logits[cache] - prune_log_sum : model_logprob;
        frontier[uint(FrontierIdx.TokenId) * frontier_capacity + slot] = winner.y;
        frontier[uint(FrontierIdx.ParentSlot) * frontier_capacity + slot] = tree_slot;
        frontier[uint(FrontierIdx.Depth) * frontier_capacity + slot] = child_depth;
        frontier[uint(FrontierIdx.PathLogprobBits) * frontier_capacity + slot] = path_bits;
        frontier[uint(FrontierIdx.EdgeLogprobBits) * frontier_capacity + slot] = asuint(edge_logprob);
        frontier[uint(FrontierIdx.PathScoreKey) * frontier_capacity + slot] = float_order_key(path_bits);
        frontier[uint(FrontierIdx.Active) * frontier_capacity + slot] = 1;
      }
      entries[cache] = neutral;
    }
  }
}

void weaver_frontier(
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
  uint tid,
  uint candidates,
  uint frontier_capacity,
  uint tree_slot_count,
  uint node_count,
  uint batch_start_slot,
  uint ancestor_stride,
  uint lookahead_count,
  uint candidate_depth_count,
) {
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
    let parent_slot = real && parent < tree_slot_count ? parent : 0;
    for (uint column = 0; column < ancestor_stride; column++) {
      uint ancestor = 0;
      if (real && column + 1 < depth) {
        ancestor = slot_ancestors[parent_slot * ancestor_stride + column];
      } else if (real && column + 1 == depth) {
        ancestor = parent_slot;
      }
      slot_ancestors[tree_slot * ancestor_stride + column] = ancestor;
      node_ancestor_indices[tid * ancestor_stride + column] = ancestor;
    }
  }

  for (uint node = 0; node < node_count; node++) {
    let slot = weaver_select_winners[node];
    let depth = slot != FRONTIER_NO_WINNER ? frontier[uint(FrontierIdx.Depth) * frontier_capacity + slot] : 0;
    if (depth >= candidate_depth_count) { continue; }
    let source = depth * candidates;
    let destination = node * candidates;
    for (uint candidate = tid; candidate < candidates; candidate += FRONTIER_SELECT_THREADS) {
      node_candidate_ids[destination + candidate] = candidate_pool_ids[source + candidate];
      node_candidate_logits[destination + candidate] = candidate_pool_logits[source + candidate];
    }
  }
}

[[Kernel]] [[Public]]
func WeaverSelect(
  [[Optional("mode != WeaverSelectionMode::Frontier")]] Ptr<bf16, Access.Read> residual_logits,
  [[Optional("mode != WeaverSelectionMode::Frontier")]] Ptr<uint, Access.Read> depth_seeds,
  Ptr<uint, Access.ReadWrite> frontier,
  Ptr<uint, Access.ReadWrite> packed_tree,
  [[Optional("mode != WeaverSelectionMode::Children")]] Ptr<uint, Access.ReadWrite> slot_ancestors,
  [[Optional("mode != WeaverSelectionMode::Children")]] Ptr<uint, Access.ReadWrite> node_token_ids,
  Ptr<uint, Access.ReadWrite> node_metadata,
  [[Optional("mode != WeaverSelectionMode::Children")]] Ptr<uint, Access.ReadWrite> node_ancestor_indices,
  Ptr<uint, Access.ReadWrite> node_valid,
  Ptr<uint, Access.Read> candidate_pool_ids,
  Ptr<uint, Access.Read> candidate_pool_logits,
  Ptr<uint, Access.ReadWrite> node_candidate_ids,
  Ptr<uint, Access.ReadWrite> node_candidate_logits,
  uint children_rows,
  uint children_from_pool,
  uint candidates,
  uint expand_width,
  uint vocab_size,
  uint frontier_capacity,
  uint tree_slot_count,
  uint node_count,
  uint batch_start_slot,
  uint ancestor_stride,
  uint lookahead_count,
  uint candidate_depth_count,
  [[Optional("has_prune_noise")]] float prune_noise_scale,
  [[Groups("if mode == WeaverSelectionMode::Children { children_rows } else { 1 }")]] uint group,
  [[Threads("256")]] uint tid,
  [[Specialize]] uint BLOCK_THREADS,
  [[Specialize]] WeaverSelectionMode mode,
  [[Specialize]] bool has_prune_noise,
) {
  static_assert(TOP_CHILDREN_THREADS == 256 && FRONTIER_SELECT_THREADS == 256, "both steps take the workgroup");
  static_assert(CANDIDATES_MAX == 2 * TOP_CHILDREN_THREADS, "two cached candidates must cover each row");
  let invocation = subgroup_invocation();
  if (mode != WeaverSelectionMode.Frontier && candidates != 0 && candidates <= CANDIDATES_MAX && expand_width != 0 &&
      expand_width <= candidates && frontier_capacity != 0 && tree_slot_count != 0) {
    for (uint step = 0; step < (mode == WeaverSelectionMode.Fused ? children_rows : 1); step++) {
      let row = mode == WeaverSelectionMode.Fused ? step : group;
      let active = row < children_rows && node_valid[row] != 0 &&
                   node_metadata[uint(MetadataIdx.TreeSlot) * children_rows + row] < tree_slot_count;
      if (!active) {
        if (mode == WeaverSelectionMode.Children) { return; }
        continue;
      }
      weaver_children(residual_logits, depth_seeds, frontier, packed_tree, node_metadata, candidate_pool_ids,
                      candidate_pool_logits, node_candidate_ids, node_candidate_logits, children_from_pool != 0,
                      children_rows, row, invocation, candidates, expand_width, vocab_size, frontier_capacity,
                      tree_slot_count, prune_noise_scale, has_prune_noise);
    }
  }
  if (mode == WeaverSelectionMode.Children) { return; }
  AllMemoryBarrierWithGroupSync();
  if (frontier_capacity == 0 || frontier_capacity > FRONTIER_MAX_SLOTS || node_count == 0 ||
      node_count > FRONTIER_MAX_WIDTH || ancestor_stride == 0 || tree_slot_count == 0 ||
      batch_start_slot + node_count > tree_slot_count || candidate_depth_count == 0 || candidates == 0) {
    return;
  }
  weaver_frontier(frontier, packed_tree, slot_ancestors, node_token_ids, node_metadata, node_ancestor_indices,
                  node_valid, candidate_pool_ids, candidate_pool_logits, node_candidate_ids, node_candidate_logits,
                  invocation, candidates, frontier_capacity, tree_slot_count, node_count, batch_start_slot,
                  ancestor_stride, lookahead_count, candidate_depth_count);
}

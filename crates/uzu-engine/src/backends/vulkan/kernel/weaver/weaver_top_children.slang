import definitions;
import common.sampling;
import common.ranking;
import common.workgroup_reduction;
import generated.weaver;

// A candidate's logit, its Gumbel selection score and its pruning noise.
float3 weaver_top_values<R : IModelFloat16>(float candidate, R residual, uint2 key, uint token, uint vocab_size) {
  let index = sampling_index(token, vocab_size);
  let word = sampling_words(key, index.x)[index.y];
  let logit = candidate + residual.toFloat();
  return float3(logit, logit + sampling_noise(float(word) * (1.0 / 4294967296.0)), sampling_noise(sampling_uniform(word)));
}

// One workgroup per row, two candidates per invocation, one workgroup argmax per child in (score, token, index) order;
// the winner's owner writes the whole frontier slot. Each U64 seed is its low/high U32 words, so no Int64 is required.
[[Kernel]] [[Public]]
func WeaverTopChildren(
  Ptr<bf16, Access.Read> residual_logits,
  Ptr<float, Access.Read> candidate_logits,
  Ptr<uint, Access.Read> candidate_ids,
  Ptr<uint, Access.Read> depth_seeds,
  Ptr<uint, Access.Read> node_metadata,
  Ptr<uint, Access.Read> packed_tree,
  Ptr<uint, Access.Read> node_valid,
  Ptr<uint, Access.ReadWrite> frontier,
  uint rows,
  uint candidates,
  uint expand_width,
  uint vocab_size,
  uint frontier_capacity,
  uint tree_slot_count,
  [[Optional("has_prune_noise")]] float prune_noise_scale,
  [[Groups("rows")]] uint row,
  [[Threads("256")]] uint lid,
  [[Specialize]] bool has_prune_noise,
) {
  static_assert(TOP_CHILDREN_THREADS == 256 && CANDIDATES_MAX == 2 * TOP_CHILDREN_THREADS, "two cached candidates must cover each row");
  if (candidates == 0 || candidates > CANDIDATES_MAX || expand_width == 0 || expand_width > candidates ||
      frontier_capacity == 0 || tree_slot_count == 0) {
    return;
  }
  if (node_valid[row] == 0) { return; }
  let tree_slot = node_metadata[uint(MetadataIdx.TreeSlot) * rows + row];
  if (tree_slot >= tree_slot_count) { return; }
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
    let token = candidate_ids[base + index];
    let values =
        weaver_top_values(candidate_logits[base + index], residual_logits[base + index], key, token, vocab_size);
    logits[cache] = values.x;
    entries[cache] = uint4(float_order_key(asuint(values.y)), token, index, index);
    if (has_prune_noise) {
      prune_logits[cache] = values.x + prune_noise_scale * values.z;
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
    let prune_maximum = workgroup_reduce<Max<float>, TOP_CHILDREN_THREADS>(max(prune_logits[0], prune_logits[1]));
    float prune_local_sum = 0.0;
    for (uint cache = 0; cache < 2; cache++) {
      if (lid + cache * TOP_CHILDREN_THREADS < candidates) {
        prune_local_sum += exp(prune_logits[cache] - prune_maximum);
      }
    }
    prune_log_sum = log(workgroup_reduce<Sum<float>, TOP_CHILDREN_THREADS>(prune_local_sum)) + prune_maximum;
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

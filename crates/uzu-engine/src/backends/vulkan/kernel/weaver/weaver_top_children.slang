import definitions;
import common.activation;
import common.sampling;
import common.storage_address;
import common.ranking;
import common.workgroup_reduction;
import generated.weaver;

groupshared uint4 weaver_top_reduce[TOP_CHILDREN_THREADS];
groupshared uint4 weaver_top_winner;

// The CPU's explicit path_logprob_bits contract for a sum's exceptional operands, on their bits: propagate signaling
// child/parent before quiet child/parent; opposite infinities give the quiet NaN. False where the sum is native: both
// operands finite, or one infinity, which the native sum keeps.
bool weaver_exceptional_sum(uint parent, uint child, out uint bits) {
  let parent_nan = (parent & 0x7fffffffu) > 0x7f800000u;
  let child_nan = (child & 0x7fffffffu) > 0x7f800000u;
  bits = 0;
  if (child_nan && (child & 0x00400000u) == 0) { bits = child | 0x00400000u; return true; }
  if (parent_nan && (parent & 0x00400000u) == 0) { bits = parent | 0x00400000u; return true; }
  if (child_nan) { bits = child | 0x00400000u; return true; }
  if (parent_nan) { bits = parent | 0x00400000u; return true; }
  if ((parent & 0x7fffffffu) == 0x7f800000u && child == (parent ^ 0x80000000u)) { bits = 0x7fc00000u; return true; }
  return false;
}

// One owner for the candidate's FP32 stages, as bits: logit, score, selection and pruning noise. Every operand is a
// loaded or computed word whose NaN is decided on its bits, never on a native sum's result: NVIDIA's compiler does not
// keep a NaN that one native sum produces through the next (a NaN logit's score came out -0).  The test entry observes
// these words before selection consumes them.
uint4 weaver_top_values<R : IModelFloat16>(float candidate, R residual, uint2 key, uint token, uint vocab_size) {
  let index = sampling_index(token, vocab_size);
  let word = sampling_words(key, index.x)[index.y];
  let noise = sampling_noise(float(word) * (1.0 / 4294967296.0));
  let target_noise = sampling_noise(sampling_uniform(word));
  uint logit;
  let residual_bits = score_bits(residual);
  if (!weaver_exceptional_sum(residual_bits, asuint(candidate), logit)) {
    logit = asuint(ordered_sum(candidate, asfloat(residual_bits)));
  }
  uint score;
  if (!weaver_exceptional_sum(asuint(noise), logit, score)) {
    score = asuint(ordered_sum(asfloat(logit), noise));
  }
  return uint4(logit, score, asuint(noise), asuint(target_noise));
}

// The CPU's explicit path_logprob_bits contract. Return integer bits so native floating-point arithmetic cannot replace
// an observable NaN payload or sign.
uint weaver_path_logprob_bits(uint parent, uint child) {
  uint bits;
  if (weaver_exceptional_sum(parent, child, bits)) {
    return bits;
  }
  return asuint(exact_sum(asfloat(parent), asfloat(child), false));
}

// Metal's fused WeaverTopChildren: one workgroup per row, two candidates per invocation, one workgroup argmax per
// child in (score, token, index) order. The winner's owner writes the whole frontier slot, where Metal splits it with
// invocation 0 re-reading the logit: one writer per word either way. Each U64 seed is its low/high U32 words, so no shader Int64 is required.
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
  let tree_slot = element_pointer(node_metadata, element_index(uint(MetadataIdx.TreeSlot), rows, row))[0];
  if (tree_slot >= tree_slot_count) { return; }
  let frontier_base = tree_slot * expand_width;
  let depth = element_pointer(node_metadata, element_index(uint(MetadataIdx.Depth), rows, row))[0];
  let seed = element_pointer(depth_seeds, element_index(depth, 2, 0));
  let key = uint2(seed[0], seed[1]);
  let neutral = uint4(0, 0xffffffffu, 0xffffffffu, 0xffffffffu);
  float logits[2];
  float prune_logits[2];
  uint4 entries[2];
  for (uint cache = 0; cache < 2; cache++) {
    let index = lid + cache * TOP_CHILDREN_THREADS;
    logits[cache] = asfloat(0xff800000u);
    prune_logits[cache] = asfloat(0xff800000u);
    entries[cache] = neutral;
    if (index >= candidates) { continue; }
    let source = element_index(row, candidates, index);
    let token = element_pointer(candidate_ids, source)[0];
    let values = weaver_top_values(element_pointer(candidate_logits, source)[0],
                                   element_pointer(residual_logits, source)[0], key, token, vocab_size);
    logits[cache] = asfloat(values.x);
    entries[cache] = uint4(float_order_key(values.y), token, index, index);
    if (has_prune_noise) {
      prune_logits[cache] = ordered_sum(asfloat(values.x), exact_product(prune_noise_scale, asfloat(values.w)));
    }
  }
  let maximum = workgroup_max<TOP_CHILDREN_THREADS>(float2(logits[0], logits[1]));
  float local_sum = 0.0;
  for (uint cache = 0; cache < 2; cache++) {
    if (lid + cache * TOP_CHILDREN_THREADS < candidates) {
      local_sum = ordered_sum(local_sum, exp(exact_sum(logits[cache], maximum, true)));
    }
  }
  let log_sum = ordered_sum(staged_log(workgroup_sum<TOP_CHILDREN_THREADS>(local_sum)), maximum);
  float prune_log_sum = 0.0;
  if (has_prune_noise) {
    let prune_maximum = workgroup_max<TOP_CHILDREN_THREADS>(float2(prune_logits[0], prune_logits[1]));
    float prune_local_sum = 0.0;
    for (uint cache = 0; cache < 2; cache++) {
      if (lid + cache * TOP_CHILDREN_THREADS < candidates) {
        prune_local_sum = ordered_sum(prune_local_sum, exp(exact_sum(prune_logits[cache], prune_maximum, true)));
      }
    }
    prune_log_sum = ordered_sum(staged_log(workgroup_sum<TOP_CHILDREN_THREADS>(prune_local_sum)), prune_maximum);
  }
  let parent_bits = element_pointer(packed_tree, element_index(uint(TreeIdx.PathLogprobBits), tree_slot_count, tree_slot))[0];
  let child_depth = element_pointer(packed_tree, element_index(uint(TreeIdx.Depth), tree_slot_count, tree_slot))[0] + 1;
  for (uint child = 0; child < expand_width; child++) {
    let partial = weaver_rank_wave(weaver_rank_better(entries[0], entries[1]));
    let lane = WavePrefixSum(1u);
    let lanes = WaveActiveSum(1u);
    if (lane == 0) { weaver_top_reduce[subgroup_id()] = partial; }
    GroupMemoryBarrierWithGroupSync();
    if (subgroup_id() == 0) {
      var total = neutral;
      for (uint subgroup = lane; subgroup < subgroup_count(); subgroup += lanes) {
        total = weaver_rank_better(total, weaver_top_reduce[subgroup]);
      }
      total = weaver_rank_wave(total);
      if (lane == 0) { weaver_top_winner = total; }
    }
    GroupMemoryBarrierWithGroupSync();
    let winner = weaver_top_winner;
    let slot = frontier_base + child;
    for (uint cache = 0; cache < 2; cache++) {
      if (lid + cache * TOP_CHILDREN_THREADS != winner.z) { continue; }
      if (slot < frontier_capacity) {
        let model_logprob = exact_sum(logits[cache], log_sum, true);
        let path_bits = weaver_path_logprob_bits(parent_bits, asuint(model_logprob));
        let edge_logprob = has_prune_noise ? exact_sum(prune_logits[cache], prune_log_sum, true) : model_logprob;
        element_pointer(frontier, element_index(uint(FrontierIdx.TokenId), frontier_capacity, slot))[0] = winner.y;
        element_pointer(frontier, element_index(uint(FrontierIdx.ParentSlot), frontier_capacity, slot))[0] = tree_slot;
        element_pointer(frontier, element_index(uint(FrontierIdx.Depth), frontier_capacity, slot))[0] = child_depth;
        element_pointer(frontier, element_index(uint(FrontierIdx.PathLogprobBits), frontier_capacity, slot))[0] = path_bits;
        element_pointer(frontier, element_index(uint(FrontierIdx.EdgeLogprobBits), frontier_capacity, slot))[0] = asuint(edge_logprob);
        element_pointer(frontier, element_index(uint(FrontierIdx.PathScoreKey), frontier_capacity, slot))[0] = float_order_key(path_bits);
        element_pointer(frontier, element_index(uint(FrontierIdx.Active), frontier_capacity, slot))[0] = 1;
      }
      entries[cache] = neutral;
    }
  }
}

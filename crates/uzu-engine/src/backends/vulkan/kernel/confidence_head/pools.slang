import definitions;
import common.sampling;
import confidence_head.head;
import generated.confidence_head;
import generated.weaver;

static const uint POOLS_THREADS = CANDIDATES_MAX;

groupshared float pools_logits[POOLS_THREADS];

[[Kernel]] [[Public]]
func ConfidenceHeadPools(
  Ptr<float, Access.Read> candidate_logits,
  Ptr<uint, Access.Read> candidate_ids,
  Ptr<uint, Access.Read> depth_seeds,
  Ptr<float, Access.ReadWrite> pool_features,
  ConfidenceHeadParams params,
  uint depths,
  [[Groups("depths")]] uint depth,
  [[Threads("512")]] uint lid,
) {
  let candidates = params.candidates;
  let invocation = subgroup_invocation();
  let valid = invocation < candidates;
  let logit = valid ? candidate_logits[depth * candidates + invocation] : NEGATIVE_INFINITY;
  pools_logits[invocation] = logit;
  let log_sum = head_log_sum_exp<POOLS_THREADS>(logit);
  if (valid) {
    var rank = 0.0;
    for (uint other = 0; other < candidates; other++) {
      rank += pools_logits[other] > logit ? 1.0 : 0.0;
    }
    let seed = uint2(depth_seeds[2 * depth], depth_seeds[2 * depth + 1]);
    let index = sampling_index(candidate_ids[depth * candidates + invocation], params.vocab_size);
    let features = pool_features + depth * POOL_FEATURES * candidates;
    features[invocation] = max(logit - log_sum, HEAD_LOG_PROB_FLOOR);
    features[candidates + invocation] = sampling_noise(sampling_uniform(sampling_words(seed, index.x)[index.y]));
    features[2 * candidates + invocation] = log(1.0 + rank);
  }
}

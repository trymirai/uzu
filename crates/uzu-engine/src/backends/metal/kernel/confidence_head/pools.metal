#include <metal_stdlib>
#include "../common/dsl.h"
#include "../rng.h"
#include "head.h"

using namespace metal;

struct PoolsMemory {
  float logits[HEAD_THREADS];
  float reduce[2u * HEAD_SIMDGROUPS];
};

PUBLIC KERNEL(ConfidenceHeadPools)(
    const device float* candidate_logits,
    const device uint* candidate_ids,
    const device uint64_t* depth_seeds,
    device float* pool_features,
    const constant ConfidenceHeadParams& params,
    const constant uint& depths,
    threadgroup PoolsMemory& memory,
    const ThreadContext context,
    const uint depth GROUPS(depths),
    const uint lid THREADS(HEAD_THREADS)
) {
  const uint candidates = params.candidates;
  const bool valid = lid < candidates;
  const float logit = valid ? candidate_logits[depth * candidates + lid] : -INFINITY;
  memory.logits[lid] = logit;
  const float log_sum = head_log_sum_exp(logit, memory.reduce, context);
  if (valid) {
    float rank = 0.0f;
    for (uint other = 0; other < candidates; ++other) {
      rank += memory.logits[other] > logit ? 1.0f : 0.0f;
    }
    device float* features = pool_features + depth * POOL_FEATURES * candidates;
    features[lid] = max(logit - log_sum, HEAD_LOG_PROB_FLOOR);
    features[candidates + lid] =
        target_gumbel_noise(depth_seeds[depth], candidate_ids[depth * candidates + lid], params.vocab_size);
    features[2 * candidates + lid] = log(1.0f + rank);
  }
}

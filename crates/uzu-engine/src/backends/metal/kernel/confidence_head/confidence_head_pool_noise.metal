#include <metal_stdlib>
#include "../common/dsl.h"
#include "confidence_head.h"

using namespace metal;

// The target sampler's Gumbel noise on every candidate of every pool row: row d draws at the seed of depth d, the
// seed the target samples the children of a depth-d node with.
PUBLIC KERNEL(ConfidenceHeadPoolNoise)(
    const device uint* candidate_ids,
    const device uint64_t* depth_seeds,
    device float* noise,
    constant uint& count,
    constant uint& candidates,
    constant uint& vocab_size,
    const uint position AXIS(count, 64)
) {
  const uint row = position / candidates;
  noise[position] = confidence_head_target_noise(depth_seeds[row], candidate_ids[position], vocab_size);
}

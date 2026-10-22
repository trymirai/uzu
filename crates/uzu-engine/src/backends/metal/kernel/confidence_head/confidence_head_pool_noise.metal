#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../rng.h"

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
  const uint token = candidate_ids[position];
  const uint thread_index = token % 1024u;
  const uint thread_offset = div_ceil(vocab_size, 1024u * 4u) * thread_index;
  const uint block_index = token / 1024u;
  PhiloxState rng;
  philox_init(&rng, depth_seeds[row], thread_offset + block_index / 4u);
  const float uniform = float(max(rng.output[block_index % 4u] >> 8, 1u)) * (1.0f / 16777216.0f);
  noise[position] = -log(-log(uniform));
}

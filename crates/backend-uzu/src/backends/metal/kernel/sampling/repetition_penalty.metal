#include <metal_stdlib>
#include "../common/dsl.h"
#include "../generated/ring.h"

using namespace metal;
using namespace uzu::ring;

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(RepetitionPenalty)(
    const device T* original_logits,
    device T* logits_copy,
    const device uint32_t* context_ring,
    const device uint64_t* token_ids,
    const constant float& repetition_penalty,
    const constant uint32_t& suffix_repetition_length,
    const uint32_t ring_index AXIS(suffix_repetition_length, 256)
) {
  const uint32_t input_token_id = static_cast<uint32_t>(token_ids[0]);
  if (ring_index == 0) {
    const float logit = static_cast<float>(original_logits[input_token_id]);
    const float penalized = logit > 0.0 ? logit / repetition_penalty : logit * repetition_penalty;
    logits_copy[input_token_id] = T(penalized);
  }

  const RingParams ring = *reinterpret_cast<const device RingParams*>(context_ring);
  if (ring_index >= ring.ring_length) {
    return;
  }

  const device uint32_t* ring_tokens = context_ring + 2;
  const uint32_t slot = (ring.ring_offset + ring_index) % suffix_repetition_length;
  const uint32_t token_id = ring_tokens[slot];
  const float logit = static_cast<float>(original_logits[token_id]);
  const float penalized = logit > 0.0 ? logit / repetition_penalty : logit * repetition_penalty;
  logits_copy[token_id] = T(penalized);
}

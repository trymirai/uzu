#include <metal_stdlib>
#include "../common/dsl.h"
#include "../generated/ring.h"
#include "../generated/trie.h"

using namespace metal;
using namespace uzu::ring;
using namespace uzu::trie;

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(RepetitionPenalty)(
    const device T* original_logits,
    device T* logits_copy,
    const device uint32_t* context_ring,
    const device uint32_t* token_ids,
    const device TrieNode* trie OPTIONAL(is_trie),
    const constant float& repetition_penalty,
    const constant uint32_t& suffix_repetition_length,
    const constant uint32_t& vocab_size,
    const constant uint32_t& sampling_start,
    const constant uint32_t& sampling_length,
    const bool is_trie SPECIALIZE,
    const uint32_t position AXIS(sampling_length * suffix_repetition_length, 256)
) {
  const RingParams ring = *reinterpret_cast<const device RingParams*>(context_ring);

  const uint32_t sample_index = position / suffix_repetition_length;
  const uint32_t window_index = position % suffix_repetition_length;
  const uint32_t batch_index = sampling_start + sample_index;
  const uint32_t batch_prefix_length = (is_trie ? trie[batch_index].height : batch_index) + 1;
  const uint32_t total_length = ring.ring_length + batch_prefix_length;
  const uint32_t window_length = min(total_length, suffix_repetition_length);
  if (window_index >= window_length) {
    return;
  }

  const uint32_t source_index = total_length - window_length + window_index;
  uint32_t token_id;
  if (source_index < ring.ring_length) {
    const device uint32_t* ring_tokens = context_ring + 2;
    const uint32_t slot = (ring.ring_offset + source_index) % suffix_repetition_length;
    token_id = ring_tokens[slot];
  } else {
    uint32_t token_index = source_index - ring.ring_length;
    if (is_trie) {
      const uint32_t height = token_index;
      token_index = batch_index;
      while (trie[token_index].height != height) {
        --token_index;
      }
    }
    token_id = token_ids[token_index];
  }

  const uint32_t logit_offset = sample_index * vocab_size + token_id;
  const float logit = static_cast<float>(original_logits[logit_offset]);
  const float penalized = logit > 0.0 ? logit / repetition_penalty : logit * repetition_penalty;
  logits_copy[logit_offset] = T(penalized);
}

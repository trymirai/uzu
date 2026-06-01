#include <metal_stdlib>
#include "../common/dsl.h"
#include "../generated/ring.h"

using namespace uzu::ring;

/// Copies a single sampled token (u32) to the token_ids buffer (u64).
/// Used in async pipeline to pass sampled token to next forward pass.
PUBLIC KERNEL(TokenCopySampled)(
    device const uint32_t* src,
    device uint64_t* dst,
    device uint32_t* context_ring OPTIONAL(has_context_ring),
    const constant uint32_t& suffix_repetition_length OPTIONAL(has_context_ring),
    const bool has_context_ring SPECIALIZE
) {
  const uint32_t token = src[0];
  dst[0] = static_cast<uint64_t>(token);

  if (has_context_ring) {
    RingParams ring = *reinterpret_cast<device RingParams*>(context_ring);
    device uint32_t* ring_tokens = context_ring + 2;
    uint32_t slot;
    if (ring.ring_length < suffix_repetition_length) {
      slot = (ring.ring_offset + ring.ring_length) % suffix_repetition_length;
      ring.ring_length += 1;
    } else {
      slot = ring.ring_offset;
      ring.ring_offset = (ring.ring_offset + 1) % suffix_repetition_length;
    }
    ring_tokens[slot] = token;
    *reinterpret_cast<device RingParams*>(context_ring) = ring;
  }
}

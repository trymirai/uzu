#include <metal_stdlib>
#include "../common/dsl.h"
#include "../generated/ring.h"

using namespace uzu::ring;

#define THREADGROUP_SIZE 256

PUBLIC KERNEL(ContextRingUpdate)(
    const device uint64_t* input,
    device uint32_t* context_ring,
    const constant uint32_t& suffix_repetition_length,
    const constant uint32_t& input_length,
    uint32_t thread_index THREADS(THREADGROUP_SIZE)
) {
  const RingParams ring = *reinterpret_cast<device RingParams*>(context_ring);

  device uint32_t* ring_tokens = context_ring + 2;
  const uint32_t write_start = input_length > suffix_repetition_length ? input_length - suffix_repetition_length : 0;

  for (uint32_t input_index = write_start + thread_index; input_index < input_length; input_index += THREADGROUP_SIZE) {
    const uint32_t slot = (ring.ring_offset + ring.ring_length + input_index) % suffix_repetition_length;
    ring_tokens[slot] = static_cast<uint32_t>(input[input_index]);
  }

  threadgroup_barrier(mem_flags::mem_none);

  if (thread_index == 0) {
    const uint32_t total_length = ring.ring_length + input_length;
    RingParams next_ring = ring;
    next_ring.ring_length = min(total_length, suffix_repetition_length);
    if (total_length > suffix_repetition_length) {
      next_ring.ring_offset = (ring.ring_offset + total_length - suffix_repetition_length) % suffix_repetition_length;
    }
    *reinterpret_cast<device RingParams*>(context_ring) = next_ring;
  }
}

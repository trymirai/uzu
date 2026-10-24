import definitions;
import generated.ring;

static const uint CONTEXT_RING_UPDATE_THREADS = 256;

// The ring header, loaded from device memory once, by invocation 0, for the whole workgroup.
groupshared RingParams context_ring_update_header;

// Stores only the last min(input_length, capacity) tokens, each to a distinct slot; invocation 0 publishes the next
// header after the barriers. The header must be valid and input must not alias the ring.
[[Kernel]] [[Public]]
[[Precondition("encode", "input_length == 0 || suffix_repetition_length != 0")]]
func ContextRingUpdate(
  Ptr<uint, Access.Read> input,
  Ptr<uint, Access.ReadWrite> context_ring,
  uint suffix_repetition_length,
  uint input_length,
  [[Groups("input_length.min(1)")]] uint group,
  [[Threads("256")]] uint tid,
) {
  static_assert(CONTEXT_RING_UPDATE_THREADS == 256, "one workgroup of Threads(256), Groups(min(1))");
  if (tid == 0) {
    context_ring_update_header.ring_offset = context_ring[0];
    context_ring_update_header.ring_length = context_ring[1];
  }
  GroupMemoryBarrierWithGroupSync();
  let ring = context_ring_update_header;
  let capacity = suffix_repetition_length;
  let write_start = input_length > capacity ? input_length - capacity : 0;
  for (uint input_index = write_start + tid; input_index < input_length; input_index += CONTEXT_RING_UPDATE_THREADS) {
    context_ring[2 + (ring.ring_offset + ring.ring_length + input_index) % capacity] = input[input_index];
  }
  GroupMemoryBarrierWithGroupSync();
  if (tid == 0) {
    let total_length = ring.ring_length + input_length;
    context_ring[1] = min(total_length, capacity);
    if (total_length > capacity) {
      context_ring[0] = (ring.ring_offset + total_length - capacity) % capacity;
    }
  }
}

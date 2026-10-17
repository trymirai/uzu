import definitions;
import common.ring_index;
import common.storage_address;
import generated.ring;

static const uint CONTEXT_RING_UPDATE_THREADS = 256;

// The ring header, loaded from device memory once, by invocation 0, for the whole workgroup.
groupshared RingParams context_ring_update_header;

// Stores only the last min(input_length, capacity) tokens, each to a distinct slot. Invocation 0 stages the two header
// words before the shared-memory barrier and publishes the next header after the second one. Header and token words
// are disjoint. The header must be valid and input must not alias the ring; productive calls require positive capacity.
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
  let capacity = suffix_repetition_length;
  uint start;
  let next = ring_append(context_ring_update_header, capacity, input_length, start);
  let count = min(input_length, capacity);
  let first = input_length - count;
  let tokens = element_pointer(context_ring, uint2(2, 0));
  // step < count <= capacity gives distinct slots; it advances only while more than one stride remains, never wrapping.
  var step = tid;
  while (step < count) {
    let slot = ring_add(start, step, capacity);
    element_pointer(tokens, uint2(slot, 0))[0] = element_pointer(input, uint2(first + step, 0))[0];
    if (count - step <= CONTEXT_RING_UPDATE_THREADS) {
      break;
    }
    step += CONTEXT_RING_UPDATE_THREADS;
  }
  GroupMemoryBarrierWithGroupSync();
  if (tid == 0) {
    context_ring[0] = next.ring_offset;
    context_ring[1] = next.ring_length;
  }
}

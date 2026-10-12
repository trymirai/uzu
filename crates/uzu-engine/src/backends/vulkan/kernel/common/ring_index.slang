import generated.ring;

// Ring arithmetic of ContextRingUpdate, shared by the kernel and its pure arithmetic test. No sum exceeds 2^32 - 1, so
// a valid ring of any positive u32 capacity takes any u32 number of appends exactly as the CPU's sequential appends.

// (a + b) mod capacity for a, b < capacity, never forming a + b.
public uint ring_add(uint a, uint b, uint capacity) {
  return a >= capacity - b ? a - (capacity - b) : a + b;
}

// `count` appends to a valid ring (capacity > 0, ring_offset < capacity, ring_length <= capacity) in closed form: the
// next header, and in `first` the slot of token count - min(count, capacity), the first one kept. count <= room is
// ring_length + count <= capacity; past it, each of the count - room appends to the full ring advances the offset.
public RingParams ring_append(RingParams ring, uint capacity, uint count, out uint first) {
  let room = capacity - ring.ring_length;
  var next = ring;
  if (count <= room) {
    next.ring_length += count;
  } else {
    next.ring_offset = ring_add(ring.ring_offset, (count - room) % capacity, capacity);
    next.ring_length = capacity;
  }
  // A full ring's length is 0 mod capacity; past capacity tokens the first kept one lands at the next offset.
  first = count > capacity ? next.ring_offset : ring_add(ring.ring_offset, room == 0 ? 0u : ring.ring_length, capacity);
  return next;
}

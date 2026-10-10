import definitions;
import common.ring_index;
import common.storage_address;
import generated.ring;

// The ring arithmetic of ContextRingUpdate on synthetic values that never form an address: for each of the `count`
// threads, cases holds ring_offset, ring_length, capacity, appended count and step of a valid ring (capacity > 0,
// ring_offset < capacity, ring_length <= capacity, step < min(count, capacity)), and results gets the first kept slot,
// the slot of that step, and the next ring_offset and ring_length.
[[Kernel]] [[Test]] func TestContextRingIndex(
  Ptr<uint, Access.Read> cases,
  Ptr<uint, Access.ReadWrite> results,
  uint count,
  [[Axis("count", "64")]] uint idx,
) {
  let c = element_pointer(cases, element_index(idx, 5, 0));
  RingParams ring;
  ring.ring_offset = c[0];
  ring.ring_length = c[1];
  let capacity = c[2];
  uint first;
  let next = ring_append(ring, capacity, c[3], first);
  let r = element_pointer(results, element_index(idx, 4, 0));
  r[0] = first;
  r[1] = ring_add(first, c[4], capacity);
  r[2] = next.ring_offset;
  r[3] = next.ring_length;
}

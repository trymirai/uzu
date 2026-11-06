import definitions;
import storage;

[[Public]] [[Kernel]] func ContextRingUpdate(
    StructuredBuffer<uint32_t> input,
    RWStructuredBuffer<uint32_t> context_ring,
    uint32_t suffix_repetition_length,
    uint32_t input_length,
    [[Threads("256")]] uint32_t thread_index,
) {
  uint lane = thread_index;

  let capacity = suffix_repetition_length;
  if (capacity == 0u) {
    return;
  }
  let offset = context_ring[0u];
  let length = context_ring[1u];
  GroupMemoryBarrierWithGroupSync();
  let skipped = input_length - min(input_length, capacity);
  for (var index = skipped + lane; index < input_length; index += 256u) {
    context_ring[2u + (offset + length + index) % capacity] = input[index];
  }
  DeviceMemoryBarrierWithGroupSync();
  if (lane == 0u) {
    let total = length + input_length;
    context_ring[0u] = (offset + total - min(total, capacity)) % capacity;
    context_ring[1u] = min(total, capacity);
  }
}

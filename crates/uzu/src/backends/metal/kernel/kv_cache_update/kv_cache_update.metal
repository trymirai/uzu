#include <metal_stdlib>
#include "../definitions.metal"

#include "kv_cache_update.h"

template <typename T>
void swap(device T* buffer, const uint sourceIdx, const uint destIdx) {
  const T temp = buffer[sourceIdx];
  buffer[sourceIdx] = buffer[destIdx];
  buffer[destIdx] = temp;
}

template <typename T>
VARIANTS(T, float, bfloat, half)
KERNEL(KVCacheUpdate) (
    device T* in_place_keys,
    device T* in_place_values,
    const constant uzu::kv_cache_update::Swap* swaps,
    const constant uint& swap_count,
    const constant uint& num_heads,
    const constant uint& max_sequence_length,
    const constant uint& head_dim,
    const uint head_idx AXIS(num_heads, 32),
    const uint channel_idx AXIS(head_dim, 32)
) {
  for (uint i = 0; i < swap_count; ++i) {
    // [headIdx: 0..num_heads, tokenIdx: 0..max_sequence_length, channelIdx:
    // 0..head_dim]
    const uint head_offset = head_idx * max_sequence_length * head_dim;
    const uint sourceIdx =
        head_offset + swaps[i].source * head_dim + channel_idx;
    const uint destIdx =
        head_offset + swaps[i].destination * head_dim + channel_idx;

    swap(in_place_keys, sourceIdx, destIdx);
    swap(in_place_values, sourceIdx, destIdx);
  }
}

#include <metal_stdlib>
#include "../common/dsl.h"

#include "kv_cache_update.h"

template <typename T>
void swap(device T* buffer, const uint sourceIdx, const uint destIdx) {
  const T temp = buffer[sourceIdx];
  buffer[sourceIdx] = buffer[destIdx];
  buffer[destIdx] = temp;
}

template <typename T>
VARIANTS(T, float, bfloat, half)
PUBLIC KERNEL(KVCacheUpdate) (
    device T* in_place_keys,
    device T* in_place_values,
    const constant uzu::kv_cache_update::Swap* swaps,
    const constant uint& swap_count,
    const constant uint& num_heads,
    const constant uint& head_dim,
    const uint head_idx AXIS(num_heads, 32),
    const uint channel_idx AXIS(head_dim, 32)
) {
  for (uint i = 0; i < swap_count; ++i) {
    // Token-major layout: [max_sequence_length, num_heads, head_dim]
    // Offset = token_idx * num_heads * head_dim +
    //          head_idx * head_dim + channel_idx
    const uint head_offset = head_idx * head_dim;
    const uint sourceIdx =
        swaps[i].source * num_heads * head_dim + head_offset + channel_idx;
    const uint destIdx =
        swaps[i].destination * num_heads * head_dim + head_offset + channel_idx;

    swap(in_place_keys, sourceIdx, destIdx);
    swap(in_place_values, sourceIdx, destIdx);
  }
}

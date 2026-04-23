#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AttentionUpdateKVCache)(
    const device T* rotated_keys OPTIONAL(!keys_in_place),
    const device T* qkv,
    device T* key_cache,
    device T* value_cache,
    const constant uint& num_groups,
    const constant uint& num_heads,
    const constant uint& head_dim,
    const constant uint& suffix_length,
    const constant uint& prefix_segment_length,
    const constant uint& max_sequence_length,
    const uint group_index AXIS(num_groups, 1),
    const uint token_index AXIS(suffix_length, 1),
    const uint dim_index AXIS(head_dim, 64),
    const bool keys_in_place SPECIALIZE
) {
  if (keys_in_place) {
    rotated_keys = key_cache;
  }

  const uint cacheTokenIndex = prefix_segment_length + token_index;

  // keys_in_place=true: destination shares rotated_keys' group-major layout.
  // Otherwise, KV cache is token-major: [max_sequence_length, num_groups, head_dim].
  const uint cacheOffset =
      keys_in_place
          ? ((group_index * max_sequence_length + cacheTokenIndex) * head_dim +
             dim_index)
          : ((cacheTokenIndex * num_groups + group_index) * head_dim +
             dim_index);

  const uint rotatedKeyOffset =
      (group_index * suffix_length + token_index) * head_dim + dim_index;
  key_cache[cacheOffset] = rotated_keys[rotatedKeyOffset];

  // qkv layout: [suffix_length, (num_heads + 2*num_groups) * head_dim]
  // Values start at offset: (num_heads * head_dim) + (num_groups * head_dim)
  const uint qkvStride = (num_heads + 2 * num_groups) * head_dim;
  const uint valueBaseOffset = (num_heads + num_groups) * head_dim;
  const uint valueOffset = token_index * qkvStride + valueBaseOffset +
                           group_index * head_dim + dim_index;
  value_cache[cacheOffset] = qkv[valueOffset];
}
#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AttentionUpdateKVCache)(
    const device T* rotated_keys,
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
    const uint dim_index AXIS(head_dim, 64)
) {
  const uint cacheTokenIndex = prefix_segment_length + token_index;

  // Copy rotated key to cache
  const uint rotatedKeyOffset =
      (group_index * suffix_length + token_index) * head_dim + dim_index;
  const uint keyCacheOffset =
      (group_index * max_sequence_length + cacheTokenIndex) * head_dim +
      dim_index;
  key_cache[keyCacheOffset] = rotated_keys[rotatedKeyOffset];

  // Copy value to cache
  // qkv layout: [suffix_length, (num_heads + 2*num_groups) * head_dim]
  // Values start at offset: (num_heads * head_dim) + (num_groups * head_dim)
  const uint qkvStride = (num_heads + 2 * num_groups) * head_dim;
  const uint valueBaseOffset = (num_heads + num_groups) * head_dim;
  const uint valueOffset = token_index * qkvStride + valueBaseOffset +
                           group_index * head_dim + dim_index;
  const uint valueCacheOffset =
      (group_index * max_sequence_length + cacheTokenIndex) * head_dim +
      dim_index;
  value_cache[valueCacheOffset] = qkv[valueOffset];
}
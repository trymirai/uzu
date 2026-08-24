#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/ring.h"
#include "../generated/trie.h"
#include "mask.h"

using namespace uzu::ring;
using namespace uzu::trie;

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AttentionFallbackScatterScores)(
    const device T* group_scores,
    device T* scores,
    const constant RingParams& ring_params OPTIONAL(is_kv_cache_ring),
    const device TrieNode* trie OPTIONAL(is_trie),
    const constant uint& sliding_window_size OPTIONAL(is_sliding_window),
    const constant uint& group_index,
    const constant uint& gqa_factor,
    const constant uint& sequence_length,
    const constant uint& suffix_length,
    const constant uint& total_elements,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_causal SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    const uint group_score_index AXIS(total_elements, 256)
) {
  const uint sequence_index = group_score_index % sequence_length;
  const uint row_index = group_score_index / sequence_length;
  const uint query_index = row_index % suffix_length;
  const uint head_in_group = row_index / suffix_length;
  const uint head_index = group_index * gqa_factor + head_in_group;

  const uint prefix_length = sequence_length - suffix_length;
  const uint suffix_position = is_kv_cache_ring ? uint(ring_params.ring_length) : prefix_length;
  const uint query_position = is_trie ? suffix_position + trie[query_index].height : suffix_position + query_index;

  const bool use_key = should_use_key(
      ring_params,
      trie,
      sliding_window_size,
      query_index,
      prefix_length,
      suffix_position,
      query_position,
      sequence_index,
      is_kv_cache_ring,
      is_causal,
      is_trie,
      is_sliding_window
  );

  scores[(head_index * suffix_length + query_index) * sequence_length + sequence_index] =
      use_key ? group_scores[group_score_index] : T(-INFINITY);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AttentionFallbackScatterValues)(
    const device T* group_output,
    device T* out,
    const constant uint& group_index,
    const constant uint& gqa_factor,
    const constant uint& suffix_length,
    const constant uint& num_heads,
    const constant uint& head_dim,
    const constant uint& total_elements,
    const uint group_output_index AXIS(total_elements, 256)
) {
  const uint dim_index = group_output_index % head_dim;
  const uint row_index = group_output_index / head_dim;
  const uint query_index = row_index % suffix_length;
  const uint head_in_group = row_index / suffix_length;
  const uint head_index = group_index * gqa_factor + head_in_group;
  out[(query_index * num_heads + head_index) * head_dim + dim_index] = group_output[group_output_index];
}

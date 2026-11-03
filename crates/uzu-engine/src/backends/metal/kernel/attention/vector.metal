#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/attention.h"
#include "kv_layout.h"
#include "mask.h"

using namespace metal;

// Adjacent queries from one KV head reuse each row of K/V codes.
template <uint HEAD_DIM, uint QUERIES>
VARIANTS(HEAD_DIM, 64, 128, 256, 512)
VARIANTS(QUERIES, 1, 2)
KERNEL(AttentionVector)(
    const device int8_t* query_codes,
    const device bfloat* query_scales,
    const device int8_t* key_codes,
    const device bfloat* key_scales,
    const device uchar* value_codes,
    const device bfloat* value_scales,
    device bfloat* output,
    device float* partial_maxs,
    device float* partial_sums,
    const device uzu::trie::TrieNode* trie OPTIONAL(is_trie),
    const constant uzu::ring::RingParams& ring_params OPTIONAL(is_kv_cache_ring),
    const constant uint& sliding_window_size OPTIONAL(is_sliding_window),
    const constant uzu::attention::AttnParams& params,
    const constant uint& rows,
    const constant uint& num_splits,
    const bool is_causal SPECIALIZE,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const uint num_kv_heads SPECIALIZE,
    const ThreadContext thread_context,
    const uint group GROUPS(rows.div_ceil(4 * QUERIES)),
    const uint split GROUPS(num_splits),
    const uint thread_index THREADS(4 * METAL_SIMD_SIZE)
) {
  const uint row = (group * 4 + thread_context.simdgroup_index) * QUERIES;
  if (row >= rows)
    return;
  constexpr uint N = (HEAD_DIM + 127) / 128;
  const ushort lane = ushort(thread_context.simd_lane_id);
  const uint key_row = lane / 4;
  const uint channel_group = lane % 4;
  const uint kv_head = row / (params.q_len * params.gqa_factor);
  const uint suffix_position = is_kv_cache_ring ? ring_params.ring_length : params.q_off;
  const uzu::kv_layout::Int8KVShape shape{HEAD_DIM, num_kv_heads, false};
  uint q_position[QUERIES];
  uint query_position[QUERIES];
  float query_scale[QUERIES];
  float4 result[QUERIES][N] = {};
  float maximum[QUERIES];
  float denominator[QUERIES] = {};
  METAL_PRAGMA_UNROLL
  for (uint q = 0; q < QUERIES; ++q) {
    q_position[q] = (row + q) % params.q_len;
    query_position[q] = suffix_position + (is_trie ? trie[q_position[q]].height : q_position[q]);
    query_scale[q] = float(query_scales[row + q]) * params.scale * M_LOG2E_F;
    maximum[q] = -INFINITY;
  }
  const uint begin = split * params.nk / num_splits;
  const uint end = (split + 1) * params.nk / num_splits;
  for (uint block = begin; block < end; ++block) {
    const size_t region = shape.block_region(block, kv_head);
    float4 scores[QUERIES] = {};
    // Keep only the current query fragment live, alongside four score sums.
    _Pragma("clang loop unroll_count(4)") for (uint fragment_index = 0; fragment_index < HEAD_DIM / 16;
                                               ++fragment_index) {
      float4 keys[4];
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 4; ++i) {
        keys[i] = float4(*reinterpret_cast<const device char4*>(
            key_codes + region + (key_row + i * 8) * HEAD_DIM + fragment_index * 16 + channel_group * 4
        ));
      }
      METAL_PRAGMA_UNROLL
      for (uint q = 0; q < QUERIES; ++q) {
        const float4 query = float4(*reinterpret_cast<const device char4*>(
            query_codes + size_t(row + q) * HEAD_DIM + fragment_index * 16 + channel_group * 4
        ));
        scores[q] += float4(dot(query, keys[0]), dot(query, keys[1]), dot(query, keys[2]), dot(query, keys[3]));
      }
    }
    float4 probabilities[QUERIES];
    float correction[QUERIES];
    METAL_PRAGMA_UNROLL
    for (uint q = 0; q < QUERIES; ++q) {
      scores[q] += simd_shuffle_xor(scores[q], ushort(1));
      scores[q] += simd_shuffle_xor(scores[q], ushort(2));
      float4 value_scale;
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 4; ++i) {
        const uint key = block * 32 + key_row + i * 8;
        const bool visible = key < params.k_len && should_use_key(
                                                       ring_params,
                                                       trie,
                                                       sliding_window_size,
                                                       q_position[q],
                                                       params.q_off,
                                                       suffix_position,
                                                       query_position[q],
                                                       key,
                                                       is_kv_cache_ring,
                                                       is_causal,
                                                       is_trie,
                                                       is_sliding_window
                                                   );
        const size_t scale_index = shape.scale_index(key, kv_head);
        scores[q][i] = visible ? scores[q][i] * query_scale[q] * float(key_scales[scale_index]) : -INFINITY;
        value_scale[i] = visible ? float(value_scales[scale_index]) : 0.0f;
      }
      const float tile_max = simd_max(max(max(scores[q].x, scores[q].y), max(scores[q].z, scores[q].w)));
      const float next_max = max(maximum[q], tile_max);
      correction[q] = denominator[q] > 0.0f ? fast::exp2(maximum[q] - next_max) : 0.0f;
      probabilities[q] = next_max == -INFINITY ? float4(0) : fast::exp2(scores[q] - next_max);
      denominator[q] =
          denominator[q] * correction[q] +
          simd_sum(probabilities[q].x + probabilities[q].y + probabilities[q].z + probabilities[q].w) * 0.25f;
      probabilities[q] *= value_scale;
      maximum[q] = next_max;
    }
    METAL_PRAGMA_UNROLL
    for (uint q = 0; q < QUERIES; ++q) {
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < N; ++i)
        result[q][i] *= correction[q];
    }
    for (uint key = 0; key < 32; ++key) {
      float weights[QUERIES];
      METAL_PRAGMA_UNROLL
      for (uint q = 0; q < QUERIES; ++q)
        weights[q] = simd_shuffle(probabilities[q][key / 8], ushort((key % 8) * 4));
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < N; ++i) {
        const uint channel = i * 128 + lane * 4;
        if (channel < HEAD_DIM) {
          const uint packed = *reinterpret_cast<const device uint*>(value_codes + region + key * HEAD_DIM + channel);
          const float4 value = float4(as_type<char4>(packed ^ 0x80808080u));
          METAL_PRAGMA_UNROLL
          for (uint q = 0; q < QUERIES; ++q)
            result[q][i] += weights[q] * value;
        }
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint q = 0; q < QUERIES; ++q) {
    const size_t partial_row = size_t(split) * rows + row + q;
    const float inverse = denominator[q] > 0.0f ? 1.0f / denominator[q] : 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < N; ++i) {
      const uint channel = i * 128 + lane * 4;
      if (channel < HEAD_DIM)
        *reinterpret_cast<device bfloat4*>(output + partial_row * HEAD_DIM + channel) = bfloat4(result[q][i] * inverse);
    }
    if (lane == 0) {
      partial_maxs[partial_row] = maximum[q];
      partial_sums[partial_row] = denominator[q];
    }
  }
}

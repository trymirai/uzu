#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/sigmoid_gate.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "../generated/attention.h"
#include "kv_layout.h"
#include "mask.h"

using namespace metal;

template <typename T, uint HEAD_DIM, uint QUERIES, bool SPLIT_KV>
VARIANTS(T, bfloat, float)
VARIANTS(HEAD_DIM, 64, 128, 256, 512)
VARIANTS(QUERIES, 1, 2)
VARIANTS(SPLIT_KV, false, true)
CONSTRAINT(SPLIT_KV || QUERIES == 1)
CONSTRAINT(!SPLIT_KV || T == "bfloat")
KERNEL(AttentionVector)(
    const device int8_t* query_codes,
    const device bfloat* query_scales,
    const device int8_t* key_codes,
    const device bfloat* key_scales,
    const device uchar* value_codes,
    const device bfloat* value_scales,
    device T* output,
    device float* partial_maxs OPTIONAL(SPLIT_KV),
    device float* partial_sums OPTIONAL(SPLIT_KV),
    const device uzu::trie::TrieNode* trie OPTIONAL(is_trie),
    const constant uzu::ring::RingParams& ring_params OPTIONAL(is_kv_cache_ring),
    const constant uint& sliding_window_size OPTIONAL(is_sliding_window),
    const device T* gate OPTIONAL(!SPLIT_KV && has_gate),
    const device T* sinks OPTIONAL(!SPLIT_KV && has_sinks),
    const constant uint& gate_row_stride OPTIONAL(!SPLIT_KV && has_gate),
    const constant uzu::attention::AttnParams& params,
    const constant uint& rows,
    const constant uint& num_splits,
    const bool is_causal SPECIALIZE,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const bool has_gate SPECIALIZE,
    const bool has_sinks SPECIALIZE,
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
  constexpr uint TILE_KEYS = SPLIT_KV ? 32 : 4;
  constexpr uint CHANNEL_LANES = SPLIT_KV ? 4 : 8;
  constexpr uint KEY_ROWS = METAL_SIMD_SIZE / CHANNEL_LANES;
  constexpr uint KEYS_PER_LANE = SPLIT_KV ? 4 : 1;
  const uint key_row = lane / CHANNEL_LANES;
  const uint channel_group = lane % CHANNEL_LANES;
  const uint kv_head = row / (params.q_len * params.gqa_factor);
  const uint suffix_position = is_kv_cache_ring ? ring_params.ring_length : params.q_off;
  const uzu::kv_layout::KVShape shape{HEAD_DIM, num_kv_heads, uzu::kv_layout::KVLayout::RowMajor};
  uint q_position[QUERIES];
  uint query_position[QUERIES];
  float query_scale[QUERIES];
  float4 result[QUERIES][N] = {};
  float maximum[QUERIES];
  float denominator[QUERIES];
  METAL_PRAGMA_UNROLL
  for (uint q = 0; q < QUERIES; ++q) {
    q_position[q] = (row + q) % params.q_len;
    query_position[q] = suffix_position + (is_trie ? trie[q_position[q]].height : q_position[q]);
    query_scale[q] = float(query_scales[row + q]) * params.scale * M_LOG2E_F;
    maximum[q] = !SPLIT_KV && has_sinks ? float(sinks[(row + q) / params.q_len]) * M_LOG2E_F : -INFINITY;
    denominator[q] = !SPLIT_KV && has_sinks ? 1.0f : 0.0f;
  }
  const uint begin = SPLIT_KV ? split * params.nk / num_splits : 0;
  const uint end = SPLIT_KV ? (split + 1) * params.nk / num_splits : (params.k_len + TILE_KEYS - 1) / TILE_KEYS;
  for (uint block = begin; block < end; ++block) {
    const uint first_key = block * TILE_KEYS;
    const size_t region = shape.block_region(first_key / 32, kv_head) + (first_key % 32) * HEAD_DIM;
    float4 scores[QUERIES] = {};
    _Pragma("clang loop unroll_count(4)") for (uint fragment_index = 0; fragment_index < HEAD_DIM / (CHANNEL_LANES * 4);
                                               ++fragment_index) {
      const uint channel = fragment_index * CHANNEL_LANES * 4 + channel_group * 4;
      float4 keys[KEYS_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < KEYS_PER_LANE; ++i) {
        keys[i] = float4(
            *reinterpret_cast<const device char4*>(key_codes + region + (key_row + i * KEY_ROWS) * HEAD_DIM + channel)
        );
      }
      METAL_PRAGMA_UNROLL
      for (uint q = 0; q < QUERIES; ++q) {
        const float4 query =
            float4(*reinterpret_cast<const device char4*>(query_codes + size_t(row + q) * HEAD_DIM + channel));
        METAL_PRAGMA_UNROLL
        for (uint i = 0; i < KEYS_PER_LANE; ++i)
          scores[q][i] += dot(query, keys[i]);
      }
    }
    float4 probabilities[QUERIES];
    float correction[QUERIES];
    METAL_PRAGMA_UNROLL
    for (uint q = 0; q < QUERIES; ++q) {
      METAL_PRAGMA_UNROLL
      for (ushort offset = 1; offset < CHANNEL_LANES; offset *= 2)
        scores[q] += simd_shuffle_xor(scores[q], offset);
      float4 value_scale = {};
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < KEYS_PER_LANE; ++i) {
        const uint key = first_key + key_row + i * KEY_ROWS;
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
      const float tile_max =
          simd_max(SPLIT_KV ? max(max(scores[q].x, scores[q].y), max(scores[q].z, scores[q].w)) : scores[q].x);
      const float next_max = max(maximum[q], tile_max);
      correction[q] = denominator[q] > 0.0f ? fast::exp2(maximum[q] - next_max) : 0.0f;
      probabilities[q] = next_max == -INFINITY ? float4(0) : fast::exp2(scores[q] - next_max);
      denominator[q] = denominator[q] * correction[q] +
                       simd_sum(
                           SPLIT_KV ? probabilities[q].x + probabilities[q].y + probabilities[q].z + probabilities[q].w
                                    : probabilities[q].x
                       ) / float(CHANNEL_LANES);
      probabilities[q] *= value_scale;
      maximum[q] = next_max;
    }
    METAL_PRAGMA_UNROLL
    for (uint q = 0; q < QUERIES; ++q) {
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < N; ++i)
        result[q][i] *= correction[q];
    }
    for (uint key = 0; key < TILE_KEYS; ++key) {
      float weights[QUERIES];
      METAL_PRAGMA_UNROLL
      for (uint q = 0; q < QUERIES; ++q)
        weights[q] = simd_shuffle(probabilities[q][key / KEY_ROWS], ushort((key % KEY_ROWS) * CHANNEL_LANES));
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
    const float inverse = denominator[q] > 0.0f ? 1.0f / denominator[q] : 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < N; ++i)
      result[q][i] *= inverse;
    if constexpr (SPLIT_KV) {
      const size_t partial_row = size_t(split) * rows + row + q;
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < N; ++i) {
        const uint channel = i * 128 + lane * 4;
        if (channel < HEAD_DIM)
          *reinterpret_cast<device bfloat4*>(output + partial_row * HEAD_DIM + channel) = bfloat4(result[q][i]);
      }
      if (lane == 0) {
        partial_maxs[partial_row] = maximum[q];
        partial_sums[partial_row] = denominator[q];
      }
    } else {
      constexpr ushort VALUES_PER_LANE = HEAD_DIM / METAL_SIMD_SIZE;
      float rotated[VALUES_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < VALUES_PER_LANE; ++i) {
        const float4 values = simd_shuffle(result[q][i / 4], ushort((i % 4) * 8 + lane / 4));
        rotated[i] = values[lane % 4];
      }
      simdgroup_hadamard_transform<VALUES_PER_LANE, METAL_SIMD_SIZE>(lane, rotated);
      const uint q_head = (row + q) / params.q_len;
      const uint output_base = q_head * params.o_strides[1] + q_position[q] * params.o_strides[2];
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < VALUES_PER_LANE; ++i) {
        const uint channel = i * METAL_SIMD_SIZE + lane;
        T value = T(rotated[i] * uzu::kv_layout::rotation_sign(channel));
        if (has_gate)
          value = uzu::apply_sigmoid_gate(value, gate[q_position[q] * gate_row_stride + q_head * HEAD_DIM + channel]);
        output[output_base + channel] = value;
      }
    }
  }
}

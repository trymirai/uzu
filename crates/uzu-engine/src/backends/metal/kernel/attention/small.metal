#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/sigmoid_gate.h"
#include "../generated/attention.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "kv_layout.h"
#include "mask.h"

using namespace metal;

// Short caches do not amortize matrix staging or a separate combine pass.
// Each SIMD group owns one query and writes its inverse-rotated output.
template <typename T, uint HEAD_DIM>
VARIANTS(T, bfloat, float)
VARIANTS(HEAD_DIM, 64, 128, 256, 512)
KERNEL(AttentionSmall)(
    const device int8_t* query_codes,
    const device bfloat* query_scales,
    const device int8_t* key_codes,
    const device bfloat* key_scales,
    const device uchar* value_codes,
    const device bfloat* value_scales,
    device T* output,
    const device uzu::trie::TrieNode* trie OPTIONAL(is_trie),
    const constant uzu::ring::RingParams& ring_params OPTIONAL(is_kv_cache_ring),
    const constant uint& sliding_window_size OPTIONAL(is_sliding_window),
    const device T* gate OPTIONAL(has_gate),
    const device T* sinks OPTIONAL(has_sinks),
    const constant uint& gate_row_stride OPTIONAL(has_gate),
    const constant uzu::attention::AttnParams& params,
    const constant uint& rows,
    const bool is_causal SPECIALIZE,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const bool has_gate SPECIALIZE,
    const bool has_sinks SPECIALIZE,
    const uint num_kv_heads SPECIALIZE,
    const ThreadContext thread_context,
    const uint group GROUPS(rows.div_ceil(4)),
    const uint thread_index THREADS(4 * METAL_SIMD_SIZE)
) {
  const uint row = group * 4 + thread_context.simdgroup_index;
  if (row >= rows)
    return;
  constexpr ushort N = HEAD_DIM / METAL_SIMD_SIZE;
  const ushort lane = ushort(thread_context.simd_lane_id);
  const uint q_head = row / params.q_len;
  const uint q_position = row % params.q_len;
  const uint kv_head = q_head / params.gqa_factor;
  const uint suffix_position = is_kv_cache_ring ? ring_params.ring_length : params.q_off;
  const uint query_position = suffix_position + (is_trie ? trie[q_position].height : q_position);
  const uzu::kv_layout::Int8KVShape shape{HEAD_DIM, num_kv_heads, false};
  const float query_scale = float(query_scales[row]) * params.scale * M_LOG2E_F;
  constexpr ushort VECTORS = (HEAD_DIM + 127) / 128;
  int4 query[VECTORS];
  float4 result[VECTORS] = {};
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < VECTORS; ++i) {
    const uint channel = i * 128 + lane * 4;
    query[i] = channel < HEAD_DIM
                   ? int4(*reinterpret_cast<const device char4*>(query_codes + size_t(row) * HEAD_DIM + channel))
                   : int4(0);
  }
  float maximum = has_sinks ? float(sinks[q_head]) * M_LOG2E_F : -INFINITY;
  float denominator = has_sinks ? 1.0f : 0.0f;
  for (uint key = 0; key < params.k_len; ++key) {
    if (!should_use_key(
            ring_params,
            trie,
            sliding_window_size,
            q_position,
            params.q_off,
            suffix_position,
            query_position,
            key,
            is_kv_cache_ring,
            is_causal,
            is_trie,
            is_sliding_window
        ))
      continue;
    int dot = 0;
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < VECTORS; ++i) {
      const uint channel = i * 128 + lane * 4;
      if (channel < HEAD_DIM) {
        const int4 key_value =
            int4(*reinterpret_cast<const device char4*>(key_codes + shape.k_code_index(key, kv_head, channel)));
        const int4 product = query[i] * key_value;
        dot += product.x + product.y + product.z + product.w;
      }
    }
    const size_t scale_index = shape.scale_index(key, kv_head);
    const float score = float(simd_sum(dot)) * query_scale * float(key_scales[scale_index]);
    const float next_max = max(maximum, score);
    const float correction = fast::exp2(maximum - next_max);
    const float probability = fast::exp2(score - next_max);
    denominator = denominator * correction + probability;
    const float weight = probability * float(value_scales[scale_index]);
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < VECTORS; ++i) {
      const uint channel = i * 128 + lane * 4;
      if (channel < HEAD_DIM) {
        const uint packed =
            *reinterpret_cast<const device uint*>(value_codes + shape.v_code_index(key, kv_head, channel));
        const float4 value = float4(as_type<char4>(packed ^ 0x80808080u));
        result[i] = result[i] * correction + weight * value;
      }
    }
    maximum = next_max;
  }
  const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < VECTORS; ++i)
    result[i] *= inverse;
  float rotated[N];
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < N; ++i) {
    const float4 values = simd_shuffle(result[i / 4], ushort((i % 4) * 8 + lane / 4));
    rotated[i] = values[lane % 4];
  }
  simdgroup_hadamard_transform<N, METAL_SIMD_SIZE>(lane, rotated);
  const uint output_base = q_head * params.o_strides[1] + q_position * params.o_strides[2];
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < N; ++i) {
    const uint channel = i * METAL_SIMD_SIZE + lane;
    T value = T(rotated[i] * uzu::kv_layout::rotation_sign(channel));
    if (has_gate)
      value = uzu::apply_sigmoid_gate(value, gate[q_position * gate_row_stride + q_head * HEAD_DIM + channel]);
    output[output_base + channel] = value;
  }
}

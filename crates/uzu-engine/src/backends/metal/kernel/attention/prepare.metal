#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "kv_layout.h"
#include "int8_quantization.h"
#include "../generated/qkv_norm.h"
#include "../generated/trie.h"

namespace kv_layout = uzu::kv_layout;

template <typename T>
METAL_FUNC T prepare_norm(
    T value,
    uint channel,
    float inverse_rms,
    const device float* scales,
    bool has_scales,
    bool enabled,
    bool full_layer,
    float scale_offset
) {
  if (!enabled)
    return value;
  const float normalized = float(value) * inverse_rms;
  if (!has_scales)
    return T(normalized);
  const float scale = scales[channel] + scale_offset;
  if (full_layer)
    return T(normalized * scale);
  return T(T(normalized) * T(scale));
}

template <typename ElementT, typename RopeT, uint HEAD_DIM>
VARIANTS(ElementT, bfloat, float)
VARIANTS(RopeT, float)
VARIANTS(HEAD_DIM, 64, 128, 256, 512)
PUBLIC KERNEL(AttentionPrepare)(
    const device ElementT* qkvg, // [token, input_row_stride], with an optional trailing gate
    device int8_t* q_codes,
    device bfloat* q_scales,
    device int8_t* k_codes OPTIONAL(append_kv),
    device bfloat* k_scales OPTIONAL(append_kv),
    device uchar* v_codes OPTIONAL(append_kv),
    device bfloat* v_scales OPTIONAL(append_kv),
    const device RopeT* cosines OPTIONAL(has_rope),
    const device RopeT* sines OPTIONAL(has_rope),
    const device float* query_norm_scales OPTIONAL(has_query_norm_scales),
    const device float* key_norm_scales OPTIONAL(has_key_norm_scales),
    const device float* value_norm_scales OPTIONAL(has_value_norm_scales),
    const constant uzu::qkv_norm::QKVNormParams& norm_params,
    const constant uint32_t& num_q_heads,
    const constant uint32_t& rope_dim OPTIONAL(has_rope),
    const constant uint32_t& kv_token_offset,
    const constant uint32_t& input_row_stride,
    const constant uint32_t& suffix_length,
    const constant uint32_t& query_rows,
    const device uzu::trie::TrieNode* trie OPTIONAL(has_trie),
    const bool has_rope SPECIALIZE,
    const uint32_t num_kv_heads SPECIALIZE,
    const uint32_t backend_opaque_layout SPECIALIZE,
    const bool append_kv SPECIALIZE,
    const bool has_query_norm_scales SPECIALIZE,
    const bool has_key_norm_scales SPECIALIZE,
    const bool has_value_norm_scales SPECIALIZE,
    const bool has_trie SPECIALIZE,
    const uint32_t token_idx GROUPS(suffix_length),
    const uint32_t head_idx GROUPS(num_q_heads + num_kv_heads * 2),
    const uint32_t lane THREADS(METAL_SIMD_SIZE)
) {
  constexpr ushort VALUES_PER_LANE = HEAD_DIM / METAL_SIMD_SIZE;
  const kv_layout::KVShape shape{HEAD_DIM, num_kv_heads, static_cast<kv_layout::KVLayout>(backend_opaque_layout)};
  const ushort lane_index = ushort(lane);
  const bool is_query = head_idx < num_q_heads;
  if (!append_kv && !is_query) {
    return;
  }
  const uint32_t gqa_factor = num_q_heads / num_kv_heads;
  const size_t query_row =
      is_query ? size_t(head_idx / gqa_factor) * query_rows + size_t(head_idx % gqa_factor) * suffix_length + token_idx
               : 0;
  const bool is_key = !is_query && head_idx < num_q_heads + num_kv_heads;
  const uint32_t kv_head = is_key ? head_idx - num_q_heads : head_idx - num_q_heads - num_kv_heads;

  const uint32_t rope_row = has_trie ? trie[token_idx].height : token_idx;
  const device ElementT* head = qkvg + size_t(token_idx) * input_row_stride + size_t(head_idx) * HEAD_DIM;
  const uint norm_index = is_query ? 0 : is_key ? 1 : 2;
  const bool norm_enabled = norm_params.enabled[norm_index];
  const bool norm_full_layer = norm_params.full_layer[norm_index];
  const float norm_scale_offset = norm_params.scale_offset[norm_index];
  const bool has_scales = is_query ? has_query_norm_scales : is_key ? has_key_norm_scales : has_value_norm_scales;
  const device float* norm_scales = nullptr;
  if (has_scales)
    norm_scales = is_query ? query_norm_scales : is_key ? key_norm_scales : value_norm_scales;
  float inverse_rms = 1.0f;
  if (norm_enabled) {
    float sum = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint base = lane * 4; base < HEAD_DIM; base += METAL_SIMD_SIZE * 4) {
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 4; ++i) {
        const float value = float(head[base + i]);
        sum += value * value;
      }
    }
    inverse_rms = rsqrt(simd_sum(sum) / float(HEAD_DIM) + norm_params.epsilon[norm_index]);
  }
  float values[VALUES_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    const uint32_t channel = value_index * METAL_SIMD_SIZE + lane_index;
    ElementT element = prepare_norm(
        head[channel],
        channel,
        inverse_rms,
        norm_scales,
        has_scales,
        norm_enabled,
        norm_full_layer,
        norm_scale_offset
    );
    if (has_rope && (is_query || is_key) && channel < rope_dim) {
      const uint half_dim = rope_dim / 2;
      const uint paired_channel = channel < half_dim ? channel + half_dim : channel - half_dim;
      const float paired = float(prepare_norm(
          head[paired_channel],
          paired_channel,
          inverse_rms,
          norm_scales,
          has_scales,
          norm_enabled,
          norm_full_layer,
          norm_scale_offset
      ));
      const float signed_paired = channel < half_dim ? -paired : paired;
      element = ElementT(
          float(element) * float(cosines[rope_row * rope_dim + channel]) +
          signed_paired * float(sines[rope_row * rope_dim + channel])
      );
    }
    values[value_index] = float(element);
  }

  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    values[value_index] *= kv_layout::rotation_sign(value_index * METAL_SIMD_SIZE + lane_index);
  }
  simdgroup_hadamard_transform<VALUES_PER_LANE, METAL_SIMD_SIZE>(lane_index, values);

  int codes[VALUES_PER_LANE];
  const bfloat scale = kv_quantize_row<VALUES_PER_LANE>(values, codes);

  const uint32_t key = kv_token_offset + token_idx;
  const size_t scale_index = is_query ? query_row : shape.scale_index(key, kv_head);
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    const uint32_t channel = value_index * METAL_SIMD_SIZE + lane_index;
    if (is_query) {
      q_codes[scale_index * HEAD_DIM + channel] = int8_t(codes[value_index]);
    } else if (is_key) {
      k_codes[shape.k_code_index(key, kv_head, channel)] = int8_t(codes[value_index]);
    } else {
      v_codes[shape.v_code_index(key, kv_head, channel)] = uchar(codes[value_index] + kv_layout::V_CODE_BIAS);
    }
  }
  if (lane_index == 0) {
    (is_query ? q_scales : is_key ? k_scales : v_scales)[scale_index] = scale;
  }
}

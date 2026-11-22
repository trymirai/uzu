#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "kv_layout.h"
#include "int8_quantization.h"
#include "../generated/attention_prepare.h"
#include "../generated/trie.h"

namespace kv_layout = uzu::kv_layout;

#define SEPARATE_PREPARE_MAX_KERNEL_SIZE 8u

// Prepares attention whose queries, keys and values project separate inputs, in one dispatch per batch:
// - queries: optional learned scales, L2 clipping and log scaling by position; then the relative logits of the head,
//   relative_logits[head][token][distance] = sum_c r[c] * profiles[c][distance], from its clipped relative states r;
// - keys and values: a causal convolution over the inputs of the token's earlier positions (its ancestors in a trie,
//   then the convolution state), with per-head dynamic taps; then L2 clipping of keys and RMS normalization of values.
// Heads are rotated, quantized to int8 and written to the query codes and the KV cache. A flat single-token batch
// shifts the convolution states in place; a trie writes the inputs each node leaves to the node inputs instead.
template <typename ElementT, uint HEAD_DIM>
VARIANTS(ElementT, bfloat, float)
VARIANTS(HEAD_DIM, 64, 128, 256)
PUBLIC KERNEL(AttentionPrepareSeparate)(
    const device ElementT* queries, // [token, query_row_stride]: queries, then the relative states at relative_offset
    const device ElementT* keys, // [token, key_row_stride]: keys, then the dynamic taps of each key head
    const device ElementT* values, // [token, value_row_stride]: values, then the dynamic taps of each value head
    const device float* key_conv_weights OPTIONAL(has_key_conv), // [kv channel, kernel_size]
    const device float* value_conv_weights OPTIONAL(has_value_conv),
    device float* key_conv_state OPTIONAL(has_key_conv), // [kv channel, kernel_size - 1], oldest first
    device float* value_conv_state OPTIONAL(has_value_conv),
    device float* key_node_inputs OPTIONAL(has_key_conv && has_trie), // [token, kv channel, kernel_size - 1]
    device float* value_node_inputs OPTIONAL(has_value_conv && has_trie),
    const device float* query_clip_scales OPTIONAL(has_query_clip_scales), // [q head, HEAD_DIM]
    const device float* relative_params OPTIONAL(has_relative_bias), // [q head, dim] clip scales, then [dim, extent]
    device bfloat* relative_logits OPTIONAL(has_relative_bias), // [q head, token, extent]
    device int8_t* q_codes,
    device bfloat* q_scales,
    device int8_t* k_codes,
    device bfloat* k_scales,
    device uchar* v_codes,
    device bfloat* v_scales,
    const device uzu::trie::TrieNode* trie OPTIONAL(has_trie),
    const constant uzu::attention_prepare::SeparatePrepareParams& params,
    const constant uint32_t& suffix_length,
    const uint32_t num_q_heads SPECIALIZE,
    const uint32_t num_kv_heads SPECIALIZE,
    const uint32_t backend_opaque_layout SPECIALIZE,
    const uint32_t kernel_size SPECIALIZE,
    const bool has_key_conv SPECIALIZE,
    const bool has_value_conv SPECIALIZE,
    const bool has_conv_taps SPECIALIZE,
    const bool has_conv_identity SPECIALIZE,
    const bool updates_conv_state SPECIALIZE,
    const bool has_query_clip SPECIALIZE,
    const bool has_query_clip_scales SPECIALIZE,
    const bool has_key_clip SPECIALIZE,
    const bool normalizes_values SPECIALIZE,
    const bool has_log_scaling SPECIALIZE,
    const bool has_relative_bias SPECIALIZE,
    const uint32_t relative_dim SPECIALIZE,
    const uint32_t relative_extent SPECIALIZE,
    const bool has_trie SPECIALIZE,
    const uint32_t token_idx GROUPS(suffix_length),
    const uint32_t head_idx GROUPS(num_q_heads + num_kv_heads * 2),
    const uint32_t lane THREADS(METAL_SIMD_SIZE)
) {
  constexpr ushort VALUES_PER_LANE = HEAD_DIM / METAL_SIMD_SIZE;
  const kv_layout::KVShape shape{HEAD_DIM, num_kv_heads, static_cast<kv_layout::KVLayout>(backend_opaque_layout)};
  const ushort lane_index = ushort(lane);
  const bool is_query = head_idx < num_q_heads;
  const bool is_key = !is_query && head_idx < num_q_heads + num_kv_heads;
  const uint32_t kv_head = is_key ? head_idx - num_q_heads : head_idx - num_q_heads - num_kv_heads;
  const uint32_t height = has_trie ? trie[token_idx].height : token_idx;
  const uint32_t kv_dim = num_kv_heads * HEAD_DIM;

  float values_[VALUES_PER_LANE];
  if (is_query) {
    const device ElementT* row = queries + size_t(token_idx) * params.query_row_stride + head_idx * HEAD_DIM;
    METAL_PRAGMA_UNROLL
    for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
      const uint channel = value_index * METAL_SIMD_SIZE + lane_index;
      values_[value_index] = float(row[channel]);
      if (has_query_clip_scales) {
        values_[value_index] *= query_clip_scales[head_idx * HEAD_DIM + channel];
      }
    }
    if (has_query_clip) {
      float sum = 0.0f;
      METAL_PRAGMA_UNROLL
      for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        sum += values_[value_index] * values_[value_index];
      }
      const float clip = metal::min(params.query_clip_threshold / (metal::sqrt(simd_sum(sum)) + 1e-6f), 1.0f);
      METAL_PRAGMA_UNROLL
      for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        values_[value_index] *= clip;
      }
    }
    float tau = 1.0f;
    if (has_log_scaling) {
      const float ratio = float(params.position_offset + height + 1) / params.log_scaling_n_floor;
      tau = 1.0f + params.log_scaling_alpha * metal::log(metal::max(ratio, 1.0f));
      METAL_PRAGMA_UNROLL
      for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        values_[value_index] *= tau;
      }
    }
    if (has_relative_bias) {
      const device float* clip_scales = relative_params;
      const device float* profiles = relative_params + num_q_heads * relative_dim;
      const device ElementT* states =
          queries + size_t(token_idx) * params.query_row_stride + params.relative_offset + head_idx * relative_dim;
      float state = lane < relative_dim ? float(states[lane]) * clip_scales[head_idx * relative_dim + lane] : 0.0f;
      const float norm = metal::sqrt(simd_sum(state * state));
      state *= metal::min(params.relative_clip_threshold / (norm + 1e-6f), 1.0f) * tau;
      // Broadcast the states once, then each lane writes 4 consecutive distances per step from vector loads of the
      // profiles. relative_dim is at most the SIMD width and relative_extent a multiple of 4.
      float broadcast[METAL_SIMD_SIZE];
      METAL_PRAGMA_UNROLL
      for (ushort channel = 0; channel < METAL_SIMD_SIZE; ++channel) {
        broadcast[channel] = channel < relative_dim ? simd_shuffle(state, channel) : 0.0f;
      }
      device bfloat* row_logits = relative_logits + (size_t(head_idx) * suffix_length + token_idx) * relative_extent;
      for (uint distance = lane * 4u; distance < relative_extent; distance += METAL_SIMD_SIZE * 4u) {
        float4 sum = float4(0.0f);
        METAL_PRAGMA_UNROLL
        for (ushort channel = 0; channel < METAL_SIMD_SIZE; ++channel) {
          if (channel < relative_dim) {
            sum += broadcast[channel] *
                   *reinterpret_cast<const device float4*>(profiles + channel * relative_extent + distance);
          }
        }
        *reinterpret_cast<device bfloat4*>(row_logits + distance) = bfloat4(sum);
      }
    }
  } else {
    const device ElementT* rows = is_key ? keys : values;
    const uint32_t row_stride = is_key ? params.key_row_stride : params.value_row_stride;
    const bool has_conv = is_key ? has_key_conv : has_value_conv;
    METAL_PRAGMA_UNROLL
    for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
      const uint channel = value_index * METAL_SIMD_SIZE + lane_index;
      values_[value_index] = float(rows[size_t(token_idx) * row_stride + kv_head * HEAD_DIM + channel]);
    }
    if (has_conv) {
      const device float* weights = is_key ? key_conv_weights : value_conv_weights;
      device float* state = is_key ? key_conv_state : value_conv_state;
      device float* node_inputs = is_key ? key_node_inputs : value_node_inputs;
      const float scale = is_key ? params.key_conv_scale : params.value_conv_scale;
      const uint32_t history = kernel_size - 1u;
      // The batch rows of the positions 1..history back: ancestors in a trie, earlier tokens otherwise.
      int sources[SEPARATE_PREPARE_MAX_KERNEL_SIZE];
      int node = int(token_idx);
      for (uint back = 1; back < kernel_size; ++back) {
        node = node < 0 ? node : (has_trie ? trie[node].parent : node - 1);
        sources[back] = node;
      }
      float taps[SEPARATE_PREPARE_MAX_KERNEL_SIZE];
      for (uint tap = 0; tap < kernel_size; ++tap) {
        taps[tap] = has_conv_taps ? float(rows[size_t(token_idx) * row_stride + kv_dim + kv_head * kernel_size + tap])
                                  : 0.0f;
      }
      METAL_PRAGMA_UNROLL
      for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        const uint channel = kv_head * HEAD_DIM + value_index * METAL_SIMD_SIZE + lane_index;
        float window[SEPARATE_PREPARE_MAX_KERNEL_SIZE];
        window[history] = values_[value_index];
        for (uint back = 1; back < kernel_size; ++back) {
          // Positions before the batch are in the state: `back` steps from a token at `height`.
          window[history - back] = sources[back] >= 0
                                       ? float(rows[size_t(sources[back]) * row_stride + channel])
                                       : state[channel * history + (history + height - back)];
        }
        float accumulator = has_conv_identity ? window[history] : 0.0f;
        for (uint tap = 0; tap < kernel_size; ++tap) {
          accumulator += (weights[channel * kernel_size + tap] + taps[tap]) * window[tap];
        }
        values_[value_index] = accumulator * scale;
        for (uint index = 0; index < history; ++index) {
          if (updates_conv_state) {
            state[channel * history + index] = window[index + 1u];
          }
          if (has_trie) {
            node_inputs[(size_t(token_idx) * kv_dim + channel) * history + index] = window[index + 1u];
          }
        }
      }
    }
    if (is_key && has_key_clip) {
      float sum = 0.0f;
      METAL_PRAGMA_UNROLL
      for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        sum += values_[value_index] * values_[value_index];
      }
      const float clip = metal::min(params.key_clip_threshold / (metal::sqrt(simd_sum(sum)) + 1e-6f), 1.0f);
      METAL_PRAGMA_UNROLL
      for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        values_[value_index] *= clip;
      }
    }
    if (!is_key && normalizes_values) {
      float sum = 0.0f;
      METAL_PRAGMA_UNROLL
      for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        sum += values_[value_index] * values_[value_index];
      }
      const float inverse_rms = rsqrt(simd_sum(sum) / float(HEAD_DIM) + params.value_norm_epsilon);
      METAL_PRAGMA_UNROLL
      for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        values_[value_index] *= inverse_rms;
      }
    }
  }

  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    values_[value_index] *= kv_layout::rotation_sign(value_index * METAL_SIMD_SIZE + lane_index);
  }
  hadamard::simdgroup_transform<VALUES_PER_LANE, METAL_SIMD_SIZE>(lane_index, values_);

  int codes[VALUES_PER_LANE];
  const bfloat scale = kv_quantize_row<VALUES_PER_LANE>(values_, codes);

  const uint32_t gqa_factor = num_q_heads / num_kv_heads;
  const size_t query_row = size_t(head_idx / gqa_factor) * params.query_rows +
                           size_t(head_idx % gqa_factor) * suffix_length + token_idx;
  const uint32_t key = params.kv_token_offset + token_idx;
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

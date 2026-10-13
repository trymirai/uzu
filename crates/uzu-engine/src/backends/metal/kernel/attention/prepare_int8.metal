#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "kv_layout.h"
#include "rope.h"

using namespace uzu::kv_layout;

UZU_CONST float INT8_MAX = 127.0f;

template <typename ElementT, ushort VALUES_PER_LANE>
METAL_FUNC ElementT kv_quantize_row(const thread float* values, thread int* codes) {
  float abs_max = 0.0f;
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    abs_max = metal::max(abs_max, metal::abs(values[value_index]));
  }
  abs_max = metal::simd_max(abs_max);
  const ElementT scale = ElementT(abs_max > 0.0f ? metal::precise::divide(abs_max, INT8_MAX) : 1.0f);
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    codes[value_index] =
        int(metal::clamp(metal::rint(metal::precise::divide(values[value_index], float(scale))), -INT8_MAX, INT8_MAX));
  }
  return scale;
}

template <typename ElementT, typename RopeT, uint HEAD_DIM>
VARIANTS(ElementT, bfloat)
VARIANTS(RopeT, float)
VARIANTS(HEAD_DIM, 256)
PUBLIC KERNEL(AttentionPrepareInt8)(
    const device ElementT* qkvg,
    device ElementT* queries OPTIONAL(!quantize_queries),
    device int8_t* q_codes OPTIONAL(quantize_queries),
    device ElementT* q_scales OPTIONAL(quantize_queries),
    device int8_t* k_codes,
    device ElementT* k_scales,
    device uchar* v_codes,
    device ElementT* v_scales,
    const device RopeT* cosines OPTIONAL(has_rope),
    const device RopeT* sines OPTIONAL(has_rope),
    const constant uint32_t& num_q_heads,
    const constant uint32_t& rope_dim OPTIONAL(has_rope),
    const constant uint32_t& kv_token_offset,
    const constant uint32_t& padded_capacity,
    const constant uint32_t& input_row_stride,
    const constant uint32_t& suffix_length,
    const constant uint32_t& padded_suffix,
    const bool has_rope SPECIALIZE,
    const bool quantize_queries SPECIALIZE,
    const uint32_t num_kv_heads SPECIALIZE,
    const uint32_t token_idx GROUPS(padded_suffix),
    const uint32_t head_idx GROUPS(num_q_heads + num_kv_heads * 2),
    const uint32_t lane THREADS(METAL_SIMD_SIZE)
) {
  constexpr ushort VALUES_PER_LANE = HEAD_DIM / METAL_SIMD_SIZE;
  const Int8KVShape shape{HEAD_DIM, num_kv_heads};
  const ushort lane_index = ushort(lane);
  const bool is_query = head_idx < num_q_heads;
  const bool is_key = !is_query && head_idx < num_q_heads + num_kv_heads;
  const uint32_t kv_head = is_key ? head_idx - num_q_heads : head_idx - num_q_heads - num_kv_heads;

  if (token_idx >= suffix_length) {
    if (is_query && quantize_queries) {
      const size_t row = size_t(head_idx) * padded_suffix + token_idx;
      METAL_PRAGMA_UNROLL
      for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        q_codes[row * HEAD_DIM + value_index * METAL_SIMD_SIZE + lane_index] = 0;
      }
      if (lane_index == 0) {
        q_scales[row] = ElementT(1.0f);
      }
    }
    return;
  }

  const device ElementT* head = qkvg + size_t(token_idx) * input_row_stride + size_t(head_idx) * HEAD_DIM;
  float values[VALUES_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    const uint32_t channel = value_index * METAL_SIMD_SIZE + lane_index;
    ElementT element = head[channel];
    if (has_rope && (is_query || is_key) && channel < rope_dim) {
      element = apply_rope(head, cosines, sines, token_idx, channel, rope_dim);
    }
    values[value_index] = float(element);
  }

  if (is_query && !quantize_queries) {
    const size_t row = size_t(head_idx) * suffix_length + token_idx;
    METAL_PRAGMA_UNROLL
    for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
      queries[row * HEAD_DIM + value_index * METAL_SIMD_SIZE + lane_index] = ElementT(values[value_index]);
    }
    return;
  }

  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    values[value_index] *= kv_rotation_sign(value_index * METAL_SIMD_SIZE + lane_index);
  }
  simdgroup_hadamard_transform<VALUES_PER_LANE, METAL_SIMD_SIZE>(lane_index, values);

  int codes[VALUES_PER_LANE];
  const ElementT scale = kv_quantize_row<ElementT, VALUES_PER_LANE>(values, codes);

  const uint32_t key = kv_token_offset + token_idx;
  const size_t scale_index =
      is_query ? size_t(head_idx) * padded_suffix + token_idx : per_token_index(key, kv_head, padded_capacity);
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    const uint32_t channel = value_index * METAL_SIMD_SIZE + lane_index;
    if (is_query) {
      q_codes[scale_index * HEAD_DIM + channel] = int8_t(codes[value_index]);
    } else if (is_key) {
      k_codes[shape.k_code_index(key, kv_head, channel)] = int8_t(codes[value_index]);
    } else {
      v_codes[shape.v_code_index(key, kv_head, channel)] = uchar(codes[value_index] + KV_V_CODE_BIAS);
    }
  }
  if (lane_index == 0) {
    (is_query ? q_scales : is_key ? k_scales : v_scales)[scale_index] = scale;
  }
}

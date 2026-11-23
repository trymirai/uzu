#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/integral_constant.h"
#include "../common/thread_context.h"
#include "../hadamard_transform/hadamard_transform.h"
#include "../weaver/weaver_frontier.h"
#include "int8_quantization.h"
#include "kv_layout.h"
#include "rope.h"

using namespace metal;

#define SIMD_GROUPS_PER_THREADGROUP 4
#define THREADS_PER_THREADGROUP (SIMD_GROUPS_PER_THREADGROUP * METAL_SIMD_SIZE)

template <uint HEAD_DIM>
VARIANTS(HEAD_DIM, 128)
PUBLIC KERNEL(AncestorAttention)(
    const device int8_t* prefix_k_codes,
    const device bfloat* prefix_k_scales,
    const device uchar* prefix_v_codes,
    const device bfloat* prefix_v_scales,
    device int8_t* node_k_codes,
    device bfloat* node_k_scales,
    device uchar* node_v_codes,
    device bfloat* node_v_scales,
    const device bfloat* current_qkv,
    const device float* rope_inverse_frequencies,
    const device uint* node_metadata,
    const device uint* ancestor_indices,
    const device uint* ancestor_counts,
    const device uint* node_indices,
    device bfloat* output,
    constant uint& rows,
    constant uint& prefix_length,
    constant uint& ancestor_stride,
    constant uint& node_capacity,
    constant uint& max_depth,
    constant float& rope_scale,
    constant float& scale,
    const uint num_heads SPECIALIZE,
    const uint32_t backend_opaque_layout SPECIALIZE,
    threadgroup float split_results[SIMD_GROUPS_PER_THREADGROUP * HEAD_DIM],
    threadgroup float split_maxs[SIMD_GROUPS_PER_THREADGROUP],
    threadgroup float split_sums[SIMD_GROUPS_PER_THREADGROUP],
    const ThreadContext thread_context,
    const uint group GROUPS(rows * num_heads),
    const uint lid THREADS(THREADS_PER_THREADGROUP)
) {
  constexpr ushort VALUES_PER_LANE = HEAD_DIM / METAL_SIMD_SIZE;
  constexpr ushort TILE_KEYS = 4;
  const uint simdgroup_index = thread_context.simdgroup_index;
  const uint row = group / num_heads;
  const uint head = group % num_heads;
  const ushort lane = ushort(thread_context.simd_lane_id);
  const uint model_dim = num_heads * HEAD_DIM;
  const uint position = node_metadata[uint(MetadataIdx::Depth) * rows + row] + 1;
  const device bfloat* qkv = current_qkv + row * 3 * model_dim + head * HEAD_DIM;
  const uzu::kv_layout::KVShape shape{
      HEAD_DIM,
      num_heads,
      static_cast<uzu::kv_layout::KVLayout>(backend_opaque_layout)
  };
  using LaneValues = vec<float, VALUES_PER_LANE>;
  using LaneCodes = vec<int, VALUES_PER_LANE>;
  const uint first_channel = lane * VALUES_PER_LANE;
  LaneValues query, key, value;
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < VALUES_PER_LANE; ++i) {
    const uint channel = first_channel + i;
    const float sign = uzu::kv_layout::rotation_sign(channel);
    query[i] = apply_rope<float>(qkv, rope_inverse_frequencies, rope_scale, position, channel, HEAD_DIM) * sign;
    if (simdgroup_index == 0) {
      key[i] =
          apply_rope<float>(qkv + model_dim, rope_inverse_frequencies, rope_scale, position, channel, HEAD_DIM) * sign;
      value[i] = float(qkv[2 * model_dim + channel]) * sign;
    }
  }
  auto quantize = [&](thread LaneValues& values, thread LaneCodes& codes) {
    hadamard::simdgroup_transform<VALUES_PER_LANE, METAL_SIMD_SIZE>(lane, reinterpret_cast<thread float*>(&values));
    return kv_quantize_row<VALUES_PER_LANE>(
        reinterpret_cast<thread float*>(&values),
        reinterpret_cast<thread int*>(&codes)
    );
  };
  LaneCodes q, k, v;
  const float score_scale = float(quantize(query, q)) * scale * M_LOG2E_F;
  bfloat k_scale, v_scale;
  if (simdgroup_index == 0) {
    k_scale = quantize(key, k);
    v_scale = quantize(value, v);
  }
  LaneValues result = 0.0f;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  const uint num_ancestors = ancestor_counts[row];
  auto load = [&](uint slot, bool is_prefix, thread float& score, thread LaneValues& values) UZU_ALWAYS_INLINE {
    const device int8_t* k_codes = is_prefix ? prefix_k_codes : node_k_codes;
    const device uchar* v_codes = is_prefix ? prefix_v_codes : node_v_codes;
    const float key_scale = float((is_prefix ? prefix_k_scales : node_k_scales)[shape.scale_index(slot, head)]);
    const float value_scale = float((is_prefix ? prefix_v_scales : node_v_scales)[shape.scale_index(slot, head)]);
    const auto key_codes = *reinterpret_cast<const device vec<char, VALUES_PER_LANE>*>(
        k_codes + shape.k_code_index(slot, head, first_channel)
    );
    const auto value_codes = *reinterpret_cast<const device vec<uchar, VALUES_PER_LANE>*>(
        v_codes + shape.v_code_index(slot, head, first_channel)
    );
    score = simd_sum(dot(LaneValues(q), LaneValues(key_codes))) * score_scale * key_scale;
    values = value_scale * (LaneValues(value_codes) - 128.0f);
  };
  auto attend = [&](thread const float* scores, thread const LaneValues* values, uint count) UZU_ALWAYS_INLINE {
    float next_max = maximum;
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < count; ++j)
      next_max = max(next_max, scores[j]);
    const float correction = fast::exp2(maximum - next_max);
    denominator *= correction;
    result *= correction;
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < count; ++j) {
      const float probability = fast::exp2(scores[j] - next_max);
      denominator += probability;
      result += probability * values[j];
    }
    maximum = next_max;
  };
  constexpr uint KEY_STRIDE = SIMD_GROUPS_PER_THREADGROUP;
  auto attend_cached = [&](uint first, uint count, bool is_prefix) UZU_ALWAYS_INLINE {
    float scores[TILE_KEYS];
    LaneValues values[TILE_KEYS];
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < count; ++j) {
      const uint index = first + j * KEY_STRIDE;
      load(is_prefix ? index : ancestor_indices[row * ancestor_stride + index], is_prefix, scores[j], values[j]);
    }
    attend(scores, values, count);
  };
  uint prefix_key = simdgroup_index;
  for (; prefix_key + (TILE_KEYS - 1) * KEY_STRIDE < prefix_length; prefix_key += TILE_KEYS * KEY_STRIDE)
    attend_cached(prefix_key, TILE_KEYS, true);
  for (; prefix_key < prefix_length; prefix_key += KEY_STRIDE)
    attend_cached(prefix_key, 1, true);
  for (uint ancestor = simdgroup_index; ancestor < num_ancestors; ancestor += KEY_STRIDE)
    attend_cached(ancestor, 1, false);
  if (simdgroup_index == 0) {
    const float score = simd_sum(dot(LaneValues(q), LaneValues(k))) * score_scale * float(k_scale);
    const LaneValues current = float(v_scale) * LaneValues(v);
    attend(&score, &current, 1);
    if (node_capacity > 0) {
      const uint node = node_indices[row];
      *reinterpret_cast<device vec<char, VALUES_PER_LANE>*>(
          node_k_codes + shape.k_code_index(node, head, first_channel)
      ) = vec<char, VALUES_PER_LANE>(k);
      *reinterpret_cast<device vec<uchar, VALUES_PER_LANE>*>(
          node_v_codes + shape.v_code_index(node, head, first_channel)
      ) = vec<uchar, VALUES_PER_LANE>(v + 128);
      if (lane == 0) {
        node_k_scales[shape.scale_index(node, head)] = k_scale;
        node_v_scales[shape.scale_index(node, head)] = v_scale;
      }
    }
  }
  hadamard::simdgroup_transform<VALUES_PER_LANE, METAL_SIMD_SIZE>(lane, reinterpret_cast<thread float*>(&result));
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < VALUES_PER_LANE; ++i)
    split_results[simdgroup_index * HEAD_DIM + first_channel + i] =
        result[i] * uzu::kv_layout::rotation_sign(first_channel + i);
  if (lane == 0) {
    split_maxs[simdgroup_index] = maximum;
    split_sums[simdgroup_index] = denominator;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float peak = -INFINITY;
  for (uint s = 0; s < SIMD_GROUPS_PER_THREADGROUP; ++s)
    peak = max(peak, split_maxs[s]);
  for (uint channel = lid; channel < HEAD_DIM; channel += THREADS_PER_THREADGROUP) {
    float merged = 0.0f;
    float total = 0.0f;
    for (uint s = 0; s < SIMD_GROUPS_PER_THREADGROUP; ++s) {
      const float correction = fast::exp2(split_maxs[s] - peak);
      total += correction * split_sums[s];
      merged += correction * split_results[s * HEAD_DIM + channel];
    }
    output[group * HEAD_DIM + channel] = bfloat(merged / total);
  }
}

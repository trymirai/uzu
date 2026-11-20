#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
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
    threadgroup float partials[SIMD_GROUPS_PER_THREADGROUP * (HEAD_DIM + 2)],
    const ThreadContext thread_context,
    const uint row_head GROUPS(rows * num_heads),
    const uint lid THREADS(THREADS_PER_THREADGROUP)
) {
  constexpr ushort N = HEAD_DIM / METAL_SIMD_SIZE;
  constexpr uint SPLITS = SIMD_GROUPS_PER_THREADGROUP;
  constexpr ushort BATCH = 4;
  const uint split = thread_context.simdgroup_index;
  const uint row = row_head / num_heads;
  const uint head = row_head % num_heads;
  const ushort lane = ushort(thread_context.simd_lane_id);
  const uint model_dim = num_heads * HEAD_DIM;
  const uint position = node_metadata[uint(MetadataIdx::Depth) * rows + row] + 1;
  const device bfloat* qkv = current_qkv + row * 3 * model_dim + head * HEAD_DIM;
  const uzu::kv_layout::KVShape shape{
      HEAD_DIM,
      num_heads,
      static_cast<uzu::kv_layout::KVLayout>(backend_opaque_layout)
  };
  float query[N], key[N], value[N];
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < N; ++i) {
    const uint channel = i * METAL_SIMD_SIZE + lane;
    const float sign = uzu::kv_layout::rotation_sign(channel);
    query[i] = apply_rope<float>(qkv, rope_inverse_frequencies, rope_scale, position, channel, HEAD_DIM) * sign;
    if (split == 0) {
      key[i] =
          apply_rope<float>(qkv + model_dim, rope_inverse_frequencies, rope_scale, position, channel, HEAD_DIM) * sign;
      value[i] = float(qkv[2 * model_dim + channel]) * sign;
    }
  }
  hadamard::simdgroup_transform<N, METAL_SIMD_SIZE>(lane, query);
  int q[N], k[N], v[N];
  const float qs = float(kv_quantize_row<N>(query, q));
  bfloat ks, vs;
  if (split == 0) {
    hadamard::simdgroup_transform<N, METAL_SIMD_SIZE>(lane, key);
    hadamard::simdgroup_transform<N, METAL_SIMD_SIZE>(lane, value);
    ks = kv_quantize_row<N>(key, k);
    vs = kv_quantize_row<N>(value, v);
  }
  float result[N] = {};
  float maximum = -INFINITY;
  float denominator = 0.0f;
  const uint num_ancestors = ancestor_counts[row];
  auto load = [&](uint slot, bool is_prefix, thread float& score, thread float* values) {
    const device int8_t* kc = is_prefix ? prefix_k_codes : node_k_codes;
    const device uchar* vc = is_prefix ? prefix_v_codes : node_v_codes;
    const float key_scale = float((is_prefix ? prefix_k_scales : node_k_scales)[shape.scale_index(slot, head)]);
    const float value_scale = float((is_prefix ? prefix_v_scales : node_v_scales)[shape.scale_index(slot, head)]);
    float dot = 0.0f;
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < N; ++i) {
      const uint channel = i * METAL_SIMD_SIZE + lane;
      dot += float(q[i]) * float(kc[shape.k_code_index(slot, head, channel)]);
      values[i] = value_scale * float(int(vc[shape.v_code_index(slot, head, channel)]) - 128);
    }
    score = simd_sum(dot) * qs * key_scale * scale;
  };
  auto attend = [&](thread const float* scores, thread const float (*values)[N], uint count) {
    float next_max = maximum;
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < count; ++j)
      next_max = max(next_max, scores[j]);
    const float correction = fast::exp(maximum - next_max);
    denominator *= correction;
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < N; ++i)
      result[i] *= correction;
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < count; ++j) {
      const float probability = fast::exp(scores[j] - next_max);
      denominator += probability;
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < N; ++i)
        result[i] += probability * values[j][i];
    }
    maximum = next_max;
  };
  auto attend_cached = [&](uint first, uint count, bool is_prefix) {
    float scores[BATCH], values[BATCH][N];
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < count; ++j) {
      const uint index = first + j * SPLITS;
      load(is_prefix ? index : ancestor_indices[row * ancestor_stride + index], is_prefix, scores[j], values[j]);
    }
    attend(scores, values, count);
  };
  uint index = split;
  for (; index + (BATCH - 1) * SPLITS < prefix_length; index += BATCH * SPLITS)
    attend_cached(index, BATCH, true);
  for (; index < prefix_length; index += SPLITS)
    attend_cached(index, 1, true);
  for (index = split; index < num_ancestors; index += SPLITS)
    attend_cached(index, 1, false);
  if (split == 0) {
    float score = 0.0f, current[N];
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < N; ++i) {
      score += float(q[i]) * float(k[i]);
      current[i] = float(vs) * float(v[i]);
    }
    score = simd_sum(score) * qs * float(ks) * scale;
    attend(&score, &current, 1);
    if (node_capacity > 0) {
      const uint node = node_indices[row];
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < N; ++i) {
        const uint channel = i * METAL_SIMD_SIZE + lane;
        node_k_codes[shape.k_code_index(node, head, channel)] = int8_t(k[i]);
        node_v_codes[shape.v_code_index(node, head, channel)] = uchar(v[i] + 128);
      }
      if (lane == 0) {
        node_k_scales[shape.scale_index(node, head)] = ks;
        node_v_scales[shape.scale_index(node, head)] = vs;
      }
    }
  }
  threadgroup float* stats = partials + SPLITS * HEAD_DIM;
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < N; ++i)
    partials[split * HEAD_DIM + i * METAL_SIMD_SIZE + lane] = result[i];
  if (lane == 0) {
    stats[split] = maximum;
    stats[SPLITS + split] = denominator;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (split != 0)
    return;
  const float peak = simd_max(lane < SPLITS ? stats[lane] : -INFINITY);
  float merged[N] = {};
  float total = 0.0f;
  for (uint s = 0; s < SPLITS; ++s) {
    const float correction = fast::exp(stats[s] - peak);
    total += correction * stats[SPLITS + s];
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < N; ++i)
      merged[i] += correction * partials[s * HEAD_DIM + i * METAL_SIMD_SIZE + lane];
  }
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < N; ++i)
    merged[i] /= total;
  hadamard::simdgroup_transform<N, METAL_SIMD_SIZE>(lane, merged);
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < N; ++i) {
    const uint channel = i * METAL_SIMD_SIZE + lane;
    output[row_head * HEAD_DIM + channel] = bfloat(merged[i] * uzu::kv_layout::rotation_sign(channel));
  }
}

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
    const device float* cosines,
    const device float* sines,
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
    constant float& scale,
    const uint num_heads SPECIALIZE,
    const uint32_t backend_opaque_layout SPECIALIZE,
    const ThreadContext thread_context,
    const uint group GROUPS((rows * num_heads).div_ceil(SIMD_GROUPS_PER_THREADGROUP)),
    const uint lid THREADS(THREADS_PER_THREADGROUP)
) {
  const uint row_head = group * SIMD_GROUPS_PER_THREADGROUP + thread_context.simdgroup_index;
  if (row_head >= rows * num_heads)
    return;
  constexpr ushort N = HEAD_DIM / METAL_SIMD_SIZE;
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
    query[i] = float(apply_rope(qkv, cosines, sines, position, channel, HEAD_DIM)) * sign;
    key[i] = float(apply_rope(qkv + model_dim, cosines, sines, position, channel, HEAD_DIM)) * sign;
    value[i] = float(qkv[2 * model_dim + channel]) * sign;
  }
  simdgroup_hadamard_transform<N, METAL_SIMD_SIZE>(lane, query);
  simdgroup_hadamard_transform<N, METAL_SIMD_SIZE>(lane, key);
  simdgroup_hadamard_transform<N, METAL_SIMD_SIZE>(lane, value);
  int q[N], k[N], v[N];
  const float qs = float(kv_quantize_row<N>(query, q));
  const bfloat ks = kv_quantize_row<N>(key, k);
  const bfloat vs = kv_quantize_row<N>(value, v);
  float result[N] = {};
  float maximum = -INFINITY;
  float denominator = 0.0f;
  const uint ancestors = ancestor_counts[row];
  const uint total = prefix_length + ancestors + 1;
  // The channel part of a KV code offset does not depend on the key: compute it once per lane.
  uint k_channel[N], v_channel[N];
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < N; ++i) {
    const uint channel = i * METAL_SIMD_SIZE + lane;
    k_channel[i] = shape.k_channel_offset(channel);
    v_channel[i] = shape.v_channel_offset(channel);
  }
  // Keys go in chunks: every load of a chunk is issued before the online-softmax recurrence consumes them in key order.
  constexpr ushort CHUNK = 2;
  for (uint base = 0; base < total; base += CHUNK) {
    int key_code[CHUNK][N], value_code[CHUNK][N];
    float key_scale[CHUNK], value_scale[CHUNK];
    METAL_PRAGMA_UNROLL
    for (ushort u = 0; u < CHUNK; ++u) {
      const uint index = base + u;
      const bool stored = index < total && index != prefix_length + ancestors;
      const bool prefix = index < prefix_length;
      const uint slot =
          !stored ? 0u : (prefix ? index : ancestor_indices[row * ancestor_stride + index - prefix_length]);
      const device int8_t* kc = (prefix ? prefix_k_codes : node_k_codes) + shape.k_key_offset(slot, head);
      const device uchar* vc = (prefix ? prefix_v_codes : node_v_codes) + shape.v_key_offset(slot, head);
      const uint scale_slot = shape.scale_index(slot, head);
      key_scale[u] = stored ? float((prefix ? prefix_k_scales : node_k_scales)[scale_slot]) : 0.0f;
      value_scale[u] = stored ? float((prefix ? prefix_v_scales : node_v_scales)[scale_slot]) : 0.0f;
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < N; ++i) {
        key_code[u][i] = stored ? int(kc[k_channel[i]]) : 0;
        value_code[u][i] = stored ? int(vc[v_channel[i]]) - 128 : 0;
      }
    }
    METAL_PRAGMA_UNROLL
    for (ushort u = 0; u < CHUNK; ++u) {
      if (base + u < total) {
        const bool own = base + u == prefix_length + ancestors;
        int dot = 0;
        METAL_PRAGMA_UNROLL
        for (ushort i = 0; i < N; ++i)
          dot += q[i] * (own ? k[i] : key_code[u][i]);
        const float score = float(simd_sum(dot)) * qs * (own ? float(ks) : key_scale[u]) * scale;
        const float next_max = max(maximum, score);
        const float correction = fast::exp(maximum - next_max);
        const float probability = fast::exp(score - next_max);
        denominator = denominator * correction + probability;
        const float weight = probability * (own ? float(vs) : value_scale[u]);
        METAL_PRAGMA_UNROLL
        for (ushort i = 0; i < N; ++i)
          result[i] = result[i] * correction + weight * float(own ? v[i] : value_code[u][i]);
        maximum = next_max;
      }
    }
  }
  if (node_capacity > 0) {
    const uint node = node_indices[row];
    const size_t k_base = shape.k_key_offset(node, head);
    const size_t v_base = shape.v_key_offset(node, head);
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < N; ++i) {
      node_k_codes[k_base + k_channel[i]] = int8_t(k[i]);
      node_v_codes[v_base + v_channel[i]] = uchar(v[i] + 128);
    }
    if (lane == 0) {
      node_k_scales[shape.scale_index(node, head)] = ks;
      node_v_scales[shape.scale_index(node, head)] = vs;
    }
  }
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < N; ++i)
    result[i] /= denominator;
  simdgroup_hadamard_transform<N, METAL_SIMD_SIZE>(lane, result);
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < N; ++i) {
    const uint channel = i * METAL_SIMD_SIZE + lane;
    output[row_head * HEAD_DIM + channel] = bfloat(result[i] * uzu::kv_layout::rotation_sign(channel));
  }
}

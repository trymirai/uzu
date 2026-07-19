#include <metal_simdgroup>
#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"

using namespace metal;

#define SIMD_GROUPS_PER_THREADGROUP 4
#define THREADS_PER_THREADGROUP (SIMD_GROUPS_PER_THREADGROUP * METAL_SIMD_SIZE)

constant uint VALUES_PER_VECTOR = 4;
constant uint QKV_COMPONENTS = 3;
constant uint KEY_COMPONENT = 1;
constant uint VALUE_COMPONENT = 2;

METAL_FUNC void attend_qkv(
    const device bfloat4* key,
    const device bfloat4* value,
    float4 query,
    thread float4& values,
    thread float& max_score,
    thread float& sum
) {
  const float score = simd_sum(dot(query, float4(*key)));
  const float new_max = max(max_score, score);
  const float old_factor = isinf(max_score) ? 0.0f : fast::exp(max_score - new_max);
  const float factor = fast::exp(score - new_max);
  sum = sum * old_factor + factor;
  values = values * old_factor + factor * float4(*value);
  max_score = new_max;
}

template <uint HEAD_DIM>
VARIANTS(HEAD_DIM, 128)
PUBLIC KERNEL(AttentionLastQuery)(
    const device bfloat* prefix_qkv,
    device bfloat* node_qkv,
    const device bfloat* current_qkv,
    const device uint* ancestor_indices,
    const device uint* ancestor_counts,
    const device uint* node_indices,
    device bfloat* output,
    constant uint& rows,
    constant uint& prefix_length,
    constant uint& ancestor_stride,
    constant float& scale,
    const uint num_heads SPECIALIZE,
    const ThreadContext thread_context,
    const uint group GROUPS((rows * num_heads).div_ceil(SIMD_GROUPS_PER_THREADGROUP)),
    const uint lid THREADS(THREADS_PER_THREADGROUP)
) {
  const uint row_head = group * SIMD_GROUPS_PER_THREADGROUP + thread_context.simdgroup_index;
  if (row_head >= rows * num_heads) {
    return;
  }

  constexpr uint vectors_per_head = HEAD_DIM / VALUES_PER_VECTOR;
  const uint row = row_head / num_heads;
  const uint head = row_head % num_heads;
  const uint lane = thread_context.simd_lane_id;
  const uint qkv_vectors = QKV_COMPONENTS * num_heads * vectors_per_head;
  const uint key_offset = (KEY_COMPONENT * num_heads + head) * vectors_per_head + lane;
  const uint value_offset = (VALUE_COMPONENT * num_heads + head) * vectors_per_head + lane;
  const device bfloat4* prefix_vectors = (const device bfloat4*)prefix_qkv;
  const device bfloat4* node_vectors = (const device bfloat4*)node_qkv;
  const device bfloat4* current_row = (const device bfloat4*)current_qkv + row * qkv_vectors;
  device bfloat4* dest_node = (device bfloat4*)node_qkv + node_indices[row] * qkv_vectors;

  const float4 query = float4(current_row[head * vectors_per_head + lane]) * scale;
  float4 values = 0;
  float max_score = -INFINITY;
  float sum = 0;
  for (uint position = 0; position < prefix_length; ++position) {
    const device bfloat4* base = prefix_vectors + position * qkv_vectors;
    attend_qkv(base + key_offset, base + value_offset, query, values, max_score, sum);
  }
  for (uint offset = 0; offset < ancestor_counts[row]; ++offset) {
    const device bfloat4* base = node_vectors + ancestor_indices[row * ancestor_stride + offset] * qkv_vectors;
    attend_qkv(base + key_offset, base + value_offset, query, values, max_score, sum);
  }
  attend_qkv(current_row + key_offset, current_row + value_offset, query, values, max_score, sum);

  dest_node[key_offset] = current_row[key_offset];
  dest_node[value_offset] = current_row[value_offset];
  ((device bfloat4*)output)[(row * num_heads + head) * vectors_per_head + lane] = bfloat4(values / sum);
}

#include <metal_simdgroup>
#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"

using namespace metal;

#define SIMD_GROUPS_PER_THREADGROUP 4
#define THREADS_PER_THREADGROUP (SIMD_GROUPS_PER_THREADGROUP * METAL_SIMD_SIZE)

constexpr uint VALUES_PER_VECTOR = 4;
constexpr uint QKV_COMPONENTS = 3;
constexpr uint KEY_COMPONENT = 1;
constexpr uint VALUE_COMPONENT = 2;

template <uint HEAD_DIM>
METAL_FUNC void attend_qkv(
    const device bfloat* qkv,
    float4 query,
    uint num_heads,
    uint head,
    uint lane,
    thread float4& values,
    thread float& max_score,
    thread float& sum
) {
  constexpr uint vectors_per_head = HEAD_DIM / VALUES_PER_VECTOR;
  const device bfloat4* key = (const device bfloat4*)qkv + (KEY_COMPONENT * num_heads + head) * vectors_per_head;
  const device bfloat4* value = (const device bfloat4*)qkv + (VALUE_COMPONENT * num_heads + head) * vectors_per_head;
  const float score = simd_sum(dot(query, float4(key[lane])));
  const float new_max = max(max_score, score);
  const float old_factor = isinf(max_score) ? 0.0f : fast::exp(max_score - new_max);
  const float factor = fast::exp(score - new_max);
  sum = sum * old_factor + factor;
  values = values * old_factor + factor * float4(value[lane]);
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
  const uint qkv_width = QKV_COMPONENTS * num_heads * HEAD_DIM;
  const device bfloat* current_row = current_qkv + row * qkv_width;
  const device bfloat4* current_row_vectors = (const device bfloat4*)current_row;
  device bfloat4* dest_node = (device bfloat4*)(node_qkv + node_indices[row] * qkv_width);

  const float4 query = float4(current_row_vectors[head * vectors_per_head + lane]) * scale;
  float4 values = 0;
  float max_score = -INFINITY;
  float sum = 0;
  for (uint position = 0; position < prefix_length; ++position) {
    attend_qkv<HEAD_DIM>(prefix_qkv + position * qkv_width, query, num_heads, head, lane, values, max_score, sum);
  }
  for (uint offset = 0; offset < ancestor_counts[row]; ++offset) {
    const uint ancestor = ancestor_indices[row * ancestor_stride + offset];
    attend_qkv<HEAD_DIM>(node_qkv + ancestor * qkv_width, query, num_heads, head, lane, values, max_score, sum);
  }
  attend_qkv<HEAD_DIM>(current_row, query, num_heads, head, lane, values, max_score, sum);

  dest_node[(KEY_COMPONENT * num_heads + head) * vectors_per_head + lane] =
      current_row_vectors[(KEY_COMPONENT * num_heads + head) * vectors_per_head + lane];
  dest_node[(VALUE_COMPONENT * num_heads + head) * vectors_per_head + lane] =
      current_row_vectors[(VALUE_COMPONENT * num_heads + head) * vectors_per_head + lane];
  ((device bfloat4*)output)[(row * num_heads + head) * vectors_per_head + lane] = bfloat4(values / sum);
}

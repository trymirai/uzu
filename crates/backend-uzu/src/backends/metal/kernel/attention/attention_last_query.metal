#include <metal_simdgroup>
#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"

using namespace metal;

#define SIMDGROUPS_PER_THREADGROUP 4
#define THREADS_PER_THREADGROUP (SIMDGROUPS_PER_THREADGROUP * METAL_SIMD_SIZE)

constant uint QKV_COMPONENTS = 3;
constant uint KEY_COMPONENT = 1;
constant uint VALUE_COMPONENT = 2;

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
  constexpr uint vectors_per_head = HEAD_DIM / 4;
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
// Packed BF16 MHA; destinations must be unique and absent from this dispatch's ancestors.
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
    const uint group GROUPS((rows * num_heads).div_ceil(SIMDGROUPS_PER_THREADGROUP)),
    const uint lid THREADS(THREADS_PER_THREADGROUP)
) {
  const uint work = group * SIMDGROUPS_PER_THREADGROUP + thread_context.simdgroup_index;
  if (work >= rows * num_heads) {
    return;
  }

  constexpr uint vectors_per_head = HEAD_DIM / 4;
  const uint row = work / num_heads;
  const uint head = work % num_heads;
  const uint lane = thread_context.simd_lane_id;
  const uint packed_width = QKV_COMPONENTS * num_heads * HEAD_DIM;
  const device bfloat* current = current_qkv + row * packed_width;
  const device bfloat4* current_vectors = (const device bfloat4*)current;
  device bfloat4* node = (device bfloat4*)(node_qkv + node_indices[row] * packed_width);

  const float4 query = float4(current_vectors[head * vectors_per_head + lane]) * scale;
  float4 values = 0;
  float max_score = -INFINITY;
  float sum = 0;
  for (uint position = 0; position < prefix_length; ++position) {
    attend_qkv<HEAD_DIM>(prefix_qkv + position * packed_width, query, num_heads, head, lane, values, max_score, sum);
  }
  for (uint offset = 0; offset < ancestor_counts[row]; ++offset) {
    const uint ancestor = ancestor_indices[row * ancestor_stride + offset];
    attend_qkv<HEAD_DIM>(node_qkv + ancestor * packed_width, query, num_heads, head, lane, values, max_score, sum);
  }
  attend_qkv<HEAD_DIM>(current, query, num_heads, head, lane, values, max_score, sum);

  node[(KEY_COMPONENT * num_heads + head) * vectors_per_head + lane] =
      current_vectors[(KEY_COMPONENT * num_heads + head) * vectors_per_head + lane];
  node[(VALUE_COMPONENT * num_heads + head) * vectors_per_head + lane] =
      current_vectors[(VALUE_COMPONENT * num_heads + head) * vectors_per_head + lane];
  ((device bfloat4*)output)[(row * num_heads + head) * vectors_per_head + lane] = bfloat4(values / sum);
}

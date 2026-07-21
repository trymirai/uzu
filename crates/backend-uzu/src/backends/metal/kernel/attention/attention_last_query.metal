#include <metal_simdgroup>
#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/integral_constant.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"

using namespace metal;

#define SIMD_GROUPS_PER_THREADGROUP 4
#define THREADS_PER_THREADGROUP (SIMD_GROUPS_PER_THREADGROUP * METAL_SIMD_SIZE)

constant uint VALUES_PER_VECTOR = 4;
constant uint QKV_COMPONENTS = 3;
constant uint KEY_COMPONENT = 1;
constant uint VALUE_COMPONENT = 2;

template <ushort UNROLL_COUNT, typename GetPtrBaseFn>
METAL_FUNC void attend_qkv(
    float4 query,
    uint key_offset,
    uint value_offset,
    thread float4& values,
    thread float& max_score,
    thread float& sum,
    GetPtrBaseFn get_ptr_base
) {
  float4 keys[UNROLL_COUNT];
  float4 position_values[UNROLL_COUNT];
  uzu::const_for_loop<0, UNROLL_COUNT, 1>([&](auto position) {
    const device bfloat4* vectors = get_ptr_base(position);
    keys[position] = float4(vectors[key_offset]);
    position_values[position] = float4(vectors[value_offset]);
  });
  float scores[UNROLL_COUNT];
  uzu::const_for_loop<0, UNROLL_COUNT, 1>([&](auto position) {
    scores[position] = simd_sum(dot(query, keys[position]));
  });
  float new_max = max_score;
  uzu::const_for_loop<0, UNROLL_COUNT, 1>([&](auto position) { new_max = max(new_max, scores[position]); });
  const float old_factor = fast::exp(max_score - new_max);
  sum *= old_factor;
  values *= old_factor;
  uzu::const_for_loop<0, UNROLL_COUNT, 1>([&](auto position) {
    const float factor = fast::exp(scores[position] - new_max);
    sum += factor;
    values += factor * position_values[position];
  });
  max_score = new_max;
}

template <uint HEAD_DIM>
VARIANTS(HEAD_DIM, 128)
PUBLIC KERNEL(AttentionLastQuery)(
    const device bfloat* prefix_qkv,
    const device bfloat* node_qkv,
    const device bfloat* current_qkv,
    const device uint* ancestor_indices,
    const device uint* ancestor_counts,
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
  constexpr ushort unroll_count = 4;
  const uint row = row_head / num_heads;
  const uint head = row_head % num_heads;
  const uint lane = thread_context.simd_lane_id;
  const uint qkv_vectors = QKV_COMPONENTS * num_heads * vectors_per_head;
  const uint key_offset = (KEY_COMPONENT * num_heads + head) * vectors_per_head + lane;
  const uint value_offset = (VALUE_COMPONENT * num_heads + head) * vectors_per_head + lane;
  const device bfloat4* prefix_vectors = (const device bfloat4*)prefix_qkv;
  const device bfloat4* node_vectors = (const device bfloat4*)node_qkv;
  const device bfloat4* current_row = (const device bfloat4*)current_qkv + row * qkv_vectors;

  const float4 query = float4(current_row[head * vectors_per_head + lane]) * scale;

  // prefix_length >= 1 (the u0 token): seed the online softmax from position 0.
  float max_score = simd_sum(dot(query, float4(prefix_vectors[key_offset])));
  float4 values = float4(prefix_vectors[value_offset]);
  float sum = 1.0f;

  uint position = 1;
  for (; position + unroll_count - 1 < prefix_length; position += unroll_count) {
    attend_qkv<unroll_count>(query, key_offset, value_offset, values, max_score, sum, [&](int step) {
      return prefix_vectors + (position + step) * qkv_vectors;
    });
  }
  for (; position < prefix_length; ++position) {
    attend_qkv<1>(query, key_offset, value_offset, values, max_score, sum, [&](int) {
      return prefix_vectors + position * qkv_vectors;
    });
  }

  const uint ancestor_count = ancestor_counts[row];
  const device uint* row_ancestors = ancestor_indices + row * ancestor_stride;
  uint offset = 0;
  for (; offset + unroll_count - 1 < ancestor_count; offset += unroll_count) {
    attend_qkv<unroll_count>(query, key_offset, value_offset, values, max_score, sum, [&](int step) {
      return node_vectors + row_ancestors[offset + step] * qkv_vectors;
    });
  }
  for (; offset < ancestor_count; ++offset) {
    attend_qkv<1>(query, key_offset, value_offset, values, max_score, sum, [&](int) {
      return node_vectors + row_ancestors[offset] * qkv_vectors;
    });
  }

  attend_qkv<1>(query, key_offset, value_offset, values, max_score, sum, [&](int) { return current_row; });

  ((device bfloat4*)output)[(row * num_heads + head) * vectors_per_head + lane] = bfloat4(values / sum);
}

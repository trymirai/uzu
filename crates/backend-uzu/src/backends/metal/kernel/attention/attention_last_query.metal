#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"

using namespace metal;

PUBLIC KERNEL(AttentionLastQuery)(
    const device bfloat* qkv,
    const device uint* lengths,
    device bfloat* output,
    constant uint& rows,
    constant uint& sequence_length,
    constant uint& num_heads,
    constant uint& head_dim,
    constant float& scale,
    threadgroup float partial_scores[4],
    threadgroup float& shared_score,
    const ThreadContext thread_context,
    const uint head GROUPS(num_heads),
    const uint row GROUPS(rows),
    const uint lid THREADS(128)
) {
  const uint length = min(lengths[row], sequence_length);
  const uint qkv_width = 3 * num_heads * head_dim;
  const device bfloat* query = qkv + ((row * sequence_length + length - 1) * qkv_width + head * head_dim);
  const uint values_per_thread = (head_dim + 127) / 128;
  float values[4] = {0, 0, 0, 0};
  float max_score = -INFINITY;
  float sum = 0;

  for (uint position = 0; position < length; ++position) {
    const device bfloat* row_qkv = qkv + (row * sequence_length + position) * qkv_width;
    const device bfloat* key = row_qkv + (num_heads + head) * head_dim;
    float partial = 0;
    for (uint item = 0; item < values_per_thread; ++item) {
      const uint dim = lid + item * 128;
      if (dim < head_dim) {
        partial += float(query[dim]) * float(key[dim]);
      }
    }
    partial = simd_sum(partial);
    if (thread_context.simd_lane_id == 0) {
      partial_scores[thread_context.simdgroup_index] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      shared_score = scale * (partial_scores[0] + partial_scores[1] + partial_scores[2] + partial_scores[3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float new_max = max(max_score, shared_score);
    const float old_factor = fast::exp(max_score - new_max);
    const float factor = fast::exp(shared_score - new_max);
    sum = sum * old_factor + factor;
    const device bfloat* value = row_qkv + (2 * num_heads + head) * head_dim;
    for (uint item = 0; item < values_per_thread; ++item) {
      const uint dim = lid + item * 128;
      if (dim < head_dim) {
        values[item] = values[item] * old_factor + factor * float(value[dim]);
      }
    }
    max_score = new_max;
  }

  for (uint item = 0; item < values_per_thread; ++item) {
    const uint dim = lid + item * 128;
    if (dim < head_dim) {
      output[(row * num_heads + head) * head_dim + dim] = bfloat(values[item] / sum);
    }
  }
}

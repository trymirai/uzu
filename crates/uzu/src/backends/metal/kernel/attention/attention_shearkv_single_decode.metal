#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"

#define PREFIX_BLOCK_SIZE 32
#define HEAD_BLOCK_SIZE 32

using namespace metal;

template <typename T, uint HEAD_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_DIM, 64, 128, 256)
PUBLIC KERNEL(AttentionShearSingleDecode)(
    const device T* queries,
    const device T* keys,
    const device uchar* packed_values,
    const device float* value_scales,
    const device float* value_biases,
    const device T* dense_values,
    device T* out,
    const constant uint& gqa_factor,
    const constant uint& prefix_length,
    const constant uint& sequence_capacity,
    const constant uint& k_head_stride,
    const constant uint& k_seq_stride,
    const constant uint& packed_value_row_bytes,
    const constant float& scale,
    const device float* sinks OPTIONAL(has_sinks),
    const constant uint& num_heads,
    const constant uint& bits,
    threadgroup float shared_max_scores[PREFIX_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    threadgroup float shared_sum_exp_scores[PREFIX_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    threadgroup float shared_outputs[PREFIX_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    const bool has_sinks SPECIALIZE,
    const ThreadContext thread_context,
    const uint head_idx GROUPS(num_heads),
    const uint tid THREADS(1024)
) {
  constexpr uint qk_elements_per_thread = HEAD_DIM / HEAD_BLOCK_SIZE;
  constexpr uint value_elements_per_thread = HEAD_DIM / HEAD_BLOCK_SIZE;
  typedef float U;

  thread U q[qk_elements_per_thread];
  thread U o[value_elements_per_thread];

  const uint kv_head_idx = head_idx / gqa_factor;
  const device T* query =
      queries + head_idx * HEAD_DIM +
      thread_context.simdgroup_index * qk_elements_per_thread;
  const device T* head_keys =
      keys + kv_head_idx * k_head_stride +
      thread_context.simdgroup_index * qk_elements_per_thread;
  const device uchar* head_values =
      packed_values + kv_head_idx * sequence_capacity * packed_value_row_bytes;
  const device float* head_scales = value_scales + kv_head_idx * sequence_capacity;
  const device float* head_biases = value_biases + kv_head_idx * sequence_capacity;
  const device T* head_dense_values =
      dense_values + kv_head_idx * sequence_capacity * HEAD_DIM +
      thread_context.simdgroup_index * value_elements_per_thread;
  device T* output =
      out + head_idx * HEAD_DIM +
      thread_context.threadgroup_index * value_elements_per_thread;

  for (uint i = 0; i < qk_elements_per_thread; i++) {
    q[i] = static_cast<U>(scale) * query[i];
  }
  for (uint i = 0; i < value_elements_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;
  if (has_sinks && thread_context.threadgroup_index == 0) {
    max_score = static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1;
  }

  const uint local_value_start =
      thread_context.simdgroup_index * value_elements_per_thread;

  for (uint token = thread_context.threadgroup_index;
       token < prefix_length;
       token += PREFIX_BLOCK_SIZE) {
    const device T* key = head_keys + token * k_seq_stride;
    U score = 0;
    for (uint i = 0; i < qk_elements_per_thread; i++) {
      score += q[i] * static_cast<U>(key[i]);
    }
    score = simd_sum(score);

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    const device uchar* row_codes = head_values + token * packed_value_row_bytes;
    const U row_scale = static_cast<U>(head_scales[token]);
    const U row_bias = static_cast<U>(head_biases[token]);
    for (uint i = 0; i < value_elements_per_thread; i++) {
      const uint dim = local_value_start + i;
      const U quantized = bits == 8
          ? static_cast<U>(row_codes[dim])
          : static_cast<U>((dim % 2 == 0) ? (row_codes[dim / 2] & 0x0F) : (row_codes[dim / 2] >> 4));
      const U decoded = row_bias + row_scale * quantized;
      o[i] = o[i] * factor + exp_score * decoded;
    }
  }

  if (prefix_length % PREFIX_BLOCK_SIZE == thread_context.threadgroup_index) {
    const device T* self_key = head_keys + prefix_length * k_seq_stride;
    U self_score = 0;
    for (uint i = 0; i < qk_elements_per_thread; i++) {
      self_score += q[i] * static_cast<U>(self_key[i]);
    }
    self_score = simd_sum(self_score);

    U new_max = max(max_score, self_score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(self_score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    for (uint i = 0; i < value_elements_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * static_cast<U>(head_dense_values[prefix_length * HEAD_DIM + i]);
    }
  }

  if (thread_context.simdgroup_index == 0) {
    shared_max_scores[thread_context.threadgroup_index] = max_score;
    shared_sum_exp_scores[thread_context.threadgroup_index] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = shared_max_scores[thread_context.simdgroup_index];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(shared_sum_exp_scores[thread_context.simdgroup_index] * factor);

  for (uint i = 0; i < value_elements_per_thread; i++) {
    shared_outputs[thread_context.simdgroup_index * HEAD_BLOCK_SIZE + thread_context.threadgroup_index] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(
               shared_outputs[thread_context.threadgroup_index * HEAD_BLOCK_SIZE + thread_context.simdgroup_index] *
               factor
           ) /
           sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (thread_context.simdgroup_index == 0) {
    for (uint i = 0; i < value_elements_per_thread; i++) {
      output[i] = static_cast<T>(o[i]);
    }
  }
}

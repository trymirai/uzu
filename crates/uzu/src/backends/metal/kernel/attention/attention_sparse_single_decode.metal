#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"

#define SELECTED_BLOCK_SIZE 32
#define HEAD_BLOCK_SIZE 32

template <typename T, uint HEAD_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_DIM, 64, 128, 256)
PUBLIC KERNEL(AttentionSparseSingleDecode)(
    const device T* queries,
    const device T* keys,
    const device T* values,
    const device int* selected_pages,
    device T* out,
    const constant uint& gqa_factor,
    const constant uint& selected_page_count,
    const constant uint& page_size,
    const constant uint& recent_start,
    const constant uint& prefix_length,
    const constant uint& k_head_stride,
    const constant uint& k_seq_stride,
    const constant uint& v_head_stride,
    const constant uint& v_seq_stride,
    const constant float& scale,
    const device float* sinks OPTIONAL(has_sinks),
    const constant uint& num_heads,
    threadgroup float shared_max_scores[SELECTED_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    threadgroup float shared_sum_exp_scores[SELECTED_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    threadgroup float shared_outputs[SELECTED_BLOCK_SIZE * HEAD_BLOCK_SIZE],
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
  const device T* head_values =
      values + kv_head_idx * v_head_stride +
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

  for (uint page_slot = 0; page_slot < selected_page_count; page_slot++) {
    const uint page_start = uint(selected_pages[page_slot]) * page_size;
    const uint page_end = page_start + page_size;
    for (uint token = page_start + thread_context.threadgroup_index;
         token < page_end;
         token += SELECTED_BLOCK_SIZE) {
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

      const device T* value = head_values + token * v_seq_stride;
      for (uint i = 0; i < value_elements_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * static_cast<U>(value[i]);
      }
    }
  }

  for (uint token = recent_start + thread_context.threadgroup_index;
       token < prefix_length;
       token += SELECTED_BLOCK_SIZE) {
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

    const device T* value = head_values + token * v_seq_stride;
    for (uint i = 0; i < value_elements_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * static_cast<U>(value[i]);
    }
  }

  if (prefix_length % SELECTED_BLOCK_SIZE == thread_context.threadgroup_index) {
    const uint token = prefix_length;
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

    const device T* value = head_values + token * v_seq_stride;
    for (uint i = 0; i < value_elements_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * static_cast<U>(value[i]);
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
  sum_exp_score =
      simd_sum(shared_sum_exp_scores[thread_context.simdgroup_index] * factor);

  for (uint i = 0; i < value_elements_per_thread; i++) {
    shared_outputs
        [thread_context.simdgroup_index * HEAD_BLOCK_SIZE +
         thread_context.threadgroup_index] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(
               shared_outputs
                   [thread_context.threadgroup_index * HEAD_BLOCK_SIZE +
                    thread_context.simdgroup_index] *
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

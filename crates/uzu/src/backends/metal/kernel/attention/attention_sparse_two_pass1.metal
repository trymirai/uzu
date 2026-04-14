#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"

#define TOTAL_BLOCKS_COUNT 32
#define SIMDGROUPS_PER_THREADGROUP 8

template <typename T, uint HEAD_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_DIM, 64, 128, 256)
PUBLIC KERNEL(AttentionSparseTwoPass1)(
    const device T* queries,
    const device T* keys,
    const device T* values,
    const device int* selected_pages,
    device float* out,
    device float* sums,
    device float* maxs,
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
    const bool has_sinks SPECIALIZE,
    const ThreadContext thread_context,
    const uint head_group_idx GROUPS(num_heads.div_ceil(SIMDGROUPS_PER_THREADGROUP)),
    const uint block_idx GROUPS(TOTAL_BLOCKS_COUNT),
    const uint tid THREADS(SIMDGROUPS_PER_THREADGROUP * METAL_SIMD_SIZE)
) {
  constexpr uint elements_per_thread = HEAD_DIM / METAL_SIMD_SIZE;

  typedef float U;

  const uint head_idx = head_group_idx * SIMDGROUPS_PER_THREADGROUP +
                        thread_context.threadgroup_index;
  if (head_idx >= num_heads) {
    return;
  }

  const uint kv_head_idx = head_idx / gqa_factor;
  thread U q[elements_per_thread];
  thread U o[elements_per_thread] = {0};

  const device T* query =
      queries + head_idx * HEAD_DIM +
      thread_context.simdgroup_index * elements_per_thread;
  const device T* head_keys =
      keys + kv_head_idx * k_head_stride +
      thread_context.simdgroup_index * elements_per_thread;
  const device T* head_values =
      values + kv_head_idx * v_head_stride +
      thread_context.simdgroup_index * elements_per_thread;
  device float* partial_out =
      out + (head_idx * TOTAL_BLOCKS_COUNT + block_idx) * HEAD_DIM +
      thread_context.simdgroup_index * elements_per_thread;
  device float* sums_out =
      sums + head_idx * TOTAL_BLOCKS_COUNT + block_idx;
  device float* maxs_out =
      maxs + head_idx * TOTAL_BLOCKS_COUNT + block_idx;

  for (uint i = 0; i < elements_per_thread; i++) {
    q[i] = static_cast<U>(scale) * query[i];
  }

  U max_score = -1e9;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1;
  }

  for (uint page_slot = 0; page_slot < selected_page_count; page_slot++) {
    const uint page_start = uint(selected_pages[page_slot]) * page_size;
    const uint page_end = page_start + page_size;
    uint token = page_start + ((block_idx + TOTAL_BLOCKS_COUNT -
                                (page_start % TOTAL_BLOCKS_COUNT)) %
                               TOTAL_BLOCKS_COUNT);
    while (token < page_end) {
      const device T* key = head_keys + token * k_seq_stride;
      U score = 0;
      for (uint i = 0; i < elements_per_thread; i++) {
        score += q[i] * static_cast<U>(key[i]);
      }
      score = simd_sum(score);

      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      const device T* value = head_values + token * v_seq_stride;
      for (uint i = 0; i < elements_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * static_cast<U>(value[i]);
      }
      token += TOTAL_BLOCKS_COUNT;
    }
  }

  uint token = recent_start + ((block_idx + TOTAL_BLOCKS_COUNT -
                                (recent_start % TOTAL_BLOCKS_COUNT)) %
                               TOTAL_BLOCKS_COUNT);
  while (token < prefix_length) {
    const device T* key = head_keys + token * k_seq_stride;
    U score = 0;
    for (uint i = 0; i < elements_per_thread; i++) {
      score += q[i] * static_cast<U>(key[i]);
    }
    score = simd_sum(score);

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    const device T* value = head_values + token * v_seq_stride;
    for (uint i = 0; i < elements_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * static_cast<U>(value[i]);
    }
    token += TOTAL_BLOCKS_COUNT;
  }

  if (prefix_length % TOTAL_BLOCKS_COUNT == block_idx) {
    const device T* key = head_keys + prefix_length * k_seq_stride;
    U score = 0;
    for (uint i = 0; i < elements_per_thread; i++) {
      score += q[i] * static_cast<U>(key[i]);
    }
    score = simd_sum(score);

    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    const device T* value = head_values + prefix_length * v_seq_stride;
    for (uint i = 0; i < elements_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * static_cast<U>(value[i]);
    }
  }

  if (thread_context.simdgroup_index == 0) {
    sums_out[0] = sum_exp_score;
    maxs_out[0] = max_score;
  }
  for (uint i = 0; i < elements_per_thread; i++) {
    partial_out[i] = o[i];
  }
}

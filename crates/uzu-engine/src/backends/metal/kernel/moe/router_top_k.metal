#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "common.h"

template <typename T, uint MAX_TOP_K>
VARIANTS(T, bfloat)
VARIANTS(MAX_TOP_K, 8)
PUBLIC KERNEL(MoeRouterTopK)(
    const device T* logits,
    const device T* biases OPTIONAL(has_biases),
    device uint* expert_ids,
    device float* expert_weights,
    const constant uint& num_experts,
    const constant uint& top_k,
    const constant uint& batch_size,
    const constant uint& logit_stride,
    const bool has_biases SPECIALIZE,
    threadgroup float shared_best_vals[MOE_ROUTER_NUM_SIMDGROUPS],
    threadgroup uint shared_best_idxs[MOE_ROUTER_NUM_SIMDGROUPS],
    const uint token_idx GROUPS(batch_size),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(MOE_ROUTER_NUM_SIMDGROUPS)
) {
  const uint tid = lane + simd_group * METAL_SIMD_SIZE;
  const device T* row = logits + token_idx * logit_stride;

  const uint slots = (num_experts + 255) / 256;
  float my_vals[MOE_ROUTER_MAX_PER_THREAD];
  uint my_ids[MOE_ROUTER_MAX_PER_THREAD];
  for (uint s = 0; s < MOE_ROUTER_MAX_PER_THREAD; ++s) {
    const uint e = tid + s * 256;
    if (e < num_experts) {
      float v = float(row[e]);
      if (has_biases) {
        v += float(biases[e]);
      }
      my_vals[s] = isnan(v) ? -INFINITY : v;
      my_ids[s] = e;
    } else {
      my_vals[s] = -INFINITY;
      my_ids[s] = 0xFFFFFFFFu;
    }
  }

  float my_chosen_val = -INFINITY;
  uint my_chosen_id = 0xFFFFFFFFu;

  for (uint k = 0; k < top_k; ++k) {
    float thread_best_val = my_vals[0];
    uint thread_best_idx = my_ids[0];
    for (uint s = 1; s < MOE_ROUTER_MAX_PER_THREAD; ++s) {
      if (s < slots) {
        const float v = my_vals[s];
        const uint id = my_ids[s];
        if (is_better_candidate(v, id, thread_best_val, thread_best_idx, num_experts)) {
          thread_best_val = v;
          thread_best_idx = id;
        }
      }
    }

    for (ushort offset = METAL_SIMD_SIZE / 2; offset > 0; offset >>= 1) {
      const float other_val = simd_shuffle_xor(thread_best_val, offset);
      const uint other_idx = simd_shuffle_xor(thread_best_idx, offset);
      if (is_better_candidate(other_val, other_idx, thread_best_val, thread_best_idx, num_experts)) {
        thread_best_val = other_val;
        thread_best_idx = other_idx;
      }
    }

    if (lane == 0) {
      shared_best_vals[simd_group] = thread_best_val;
      shared_best_idxs[simd_group] = thread_best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
      float v = (lane < MOE_ROUTER_NUM_SIMDGROUPS) ? shared_best_vals[lane] : -INFINITY;
      uint id = (lane < MOE_ROUTER_NUM_SIMDGROUPS) ? shared_best_idxs[lane] : 0xFFFFFFFFu;
      for (ushort offset = 4; offset > 0; offset >>= 1) {
        const float other_val = simd_shuffle_xor(v, offset);
        const uint other_id = simd_shuffle_xor(id, offset);
        if (is_better_candidate(other_val, other_id, v, id, num_experts)) {
          v = other_val;
          id = other_id;
        }
      }
      if (lane == k) {
        my_chosen_val = (id < num_experts) ? v : -INFINITY;
        my_chosen_id = (id < num_experts) ? id : min(k, num_experts - 1);
      }
      if (lane == 0) {
        shared_best_idxs[0] = (id < num_experts) ? id : min(k, num_experts - 1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint winner_id = shared_best_idxs[0];
    if (winner_id % 256 == tid) {
      const uint s = winner_id / 256;
      if (s < MOE_ROUTER_MAX_PER_THREAD) {
        my_vals[s] = -INFINITY;
        my_ids[s] = 0xFFFFFFFFu;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_group == 0) {
    const float max_chosen = simd_shuffle(my_chosen_val, 0);

    const float e_val =
        (lane < top_k) ? ((max_chosen == -INFINITY) ? (1.0f / float(top_k)) : fast::exp(my_chosen_val - max_chosen))
                       : 0.0f;

    float sum_exp = 0.0f;
    for (uint i = 0; i < top_k; ++i) {
      sum_exp += simd_shuffle(e_val, i);
    }

    if (lane < top_k) {
      const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
      device uint* out_ids = expert_ids + token_idx * top_k;
      device float* out_weights = expert_weights + token_idx * top_k;
      out_ids[lane] = my_chosen_id;
      out_weights[lane] = e_val * inv_sum;
    }
  }
}

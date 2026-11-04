#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/top_k.h"
#include "common.h"

// Requires num_experts <= METAL_SIMD_SIZE * MOE_ROUTER_NUM_SIMDGROUPS, 1 <= top_k <= min(MAX_TOP_K, num_experts) and
// logit_stride >= num_experts: token token_idx's logits start at logits[token_idx * logit_stride].
template <typename T, uint MAX_TOP_K>
VARIANTS(T, float)
VARIANTS(MAX_TOP_K, 8)
PUBLIC KERNEL(MoeRouterTopK)(
    const device T* logits,
    device uint* expert_ids,
    device float* expert_weights,
    const constant uint& num_experts,
    const constant uint& top_k,
    const constant uint& batch_size,
    const constant uint& logit_stride,
    threadgroup uint simd_group_best_keys[MOE_ROUTER_NUM_SIMDGROUPS],
    threadgroup uint simd_group_best_ids[MOE_ROUTER_NUM_SIMDGROUPS],
    const uint token_idx GROUPS(batch_size),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(MOE_ROUTER_NUM_SIMDGROUPS)
) {
  static_assert(METAL_SIMD_SIZE % MOE_ROUTER_NUM_SIMDGROUPS == 0, "one SIMD group merges the group winners");

  const device T* row = logits + token_idx * logit_stride;
  // Each thread holds one expert. Key 0 ranks below the key of every logit, -INFINITY included, so neither an expert
  // past num_experts nor an expert already picked can become a round's pick.
  const uint expert_idx = lane + simd_group * METAL_SIMD_SIZE;
  uint key = 0;
  if (expert_idx < num_experts) {
    const float logit = float(row[expert_idx]);
    key = top_k_score_key(isnan(logit) ? -INFINITY : logit);
  }
  bool picked = false;

  float chosen_vals[MAX_TOP_K];
  uint chosen_ids[MAX_TOP_K];

  for (uint k = 0; k < top_k; ++k) {
    // Each SIMD group finds its best expert not picked yet.
    uint best_key = picked ? 0 : key;
    uint best_idx = expert_idx;
    for (ushort offset = METAL_SIMD_SIZE / 2; offset > 0; offset >>= 1) {
      const uint other_key = simd_shuffle_xor(best_key, offset);
      const uint other_idx = simd_shuffle_xor(best_idx, offset);
      if (other_key > best_key || (other_key == best_key && other_idx < best_idx)) {
        best_key = other_key;
        best_idx = other_idx;
      }
    }
    if (lane == 0) {
      simd_group_best_keys[simd_group] = best_key;
      simd_group_best_ids[simd_group] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD group 0 merges the group winners in its first MOE_ROUTER_NUM_SIMDGROUPS lanes, which the butterfly keeps
    // apart from the other lanes, and publishes the pick in simd_group_best_ids[0].
    if (simd_group == 0) {
      if (lane < MOE_ROUTER_NUM_SIMDGROUPS) {
        best_key = simd_group_best_keys[lane];
        best_idx = simd_group_best_ids[lane];
      }
      for (ushort offset = MOE_ROUTER_NUM_SIMDGROUPS / 2; offset > 0; offset >>= 1) {
        const uint other_key = simd_shuffle_xor(best_key, offset);
        const uint other_idx = simd_shuffle_xor(best_idx, offset);
        if (other_key > best_key || (other_key == best_key && other_idx < best_idx)) {
          best_key = other_key;
          best_idx = other_idx;
        }
      }
      if (lane == 0) {
        simd_group_best_ids[0] = best_idx;
        chosen_vals[k] = top_k_score_from_key(best_key);
        chosen_ids[k] = best_idx;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The thread that holds the pick marks it. The barrier keeps the next round from overwriting the pick before every
    // thread has read it.
    if (simd_group_best_ids[0] == expert_idx) {
      picked = true;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_group == 0 && lane == 0) {
    const float max_chosen = chosen_vals[0]; // the picks are descending
    float sum_exp = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
      const float e_val = (max_chosen == -INFINITY) ? (1.0f / float(top_k)) : fast::exp(chosen_vals[k] - max_chosen);
      chosen_vals[k] = e_val;
      sum_exp += e_val;
    }
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    device uint* out_ids = expert_ids + token_idx * top_k;
    device float* out_weights = expert_weights + token_idx * top_k;
    for (uint k = 0; k < top_k; ++k) {
      out_ids[k] = chosen_ids[k];
      out_weights[k] = chosen_vals[k] * inv_sum;
    }
  }
}

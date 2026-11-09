#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/top_k.h"
#include "common.h"

static_assert(METAL_SIMD_SIZE * MOE_ROUTER_MAX_PER_LANE == uzu::moe::ROUTER_MAX_EXPERTS);

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
    const uint token_idx GROUPS(batch_size),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  const device T* row = logits + token_idx * num_experts;
  uint keys[MOE_ROUTER_MAX_PER_LANE];
  for (uint i = 0; i < MOE_ROUTER_MAX_PER_LANE; ++i) {
    const uint expert_idx = lane + i * METAL_SIMD_SIZE;
    if (expert_idx < num_experts) {
      const float logit = float(row[expert_idx]);
      keys[i] = top_k_score_key(isnan(logit) ? -INFINITY : logit);
    } else {
      keys[i] = 0;
    }
  }

  float chosen_vals[MAX_TOP_K];
  uint chosen_ids[MAX_TOP_K];
  // Bit i is set once expert lane + i * METAL_SIMD_SIZE is picked.
  uint picked = 0;
  static_assert(MOE_ROUTER_MAX_PER_LANE <= 8 * sizeof(picked), "picked holds one bit per slot");

  for (uint k = 0; k < top_k; ++k) {
    uint best_key = 0;
    uint best_idx = 0xFFFFFFFFu;
    for (uint i = 0; i < MOE_ROUTER_MAX_PER_LANE; ++i) {
      const uint expert_idx = lane + i * METAL_SIMD_SIZE;
      if (expert_idx < num_experts && (picked & (1u << i)) == 0) {
        if (keys[i] > best_key || (keys[i] == best_key && expert_idx < best_idx)) {
          best_key = keys[i];
          best_idx = expert_idx;
        }
      }
    }

    for (ushort offset = METAL_SIMD_SIZE / 2; offset > 0; offset >>= 1) {
      const uint other_key = simd_shuffle_xor(best_key, offset);
      const uint other_idx = simd_shuffle_xor(best_idx, offset);
      if (other_key > best_key || (other_key == best_key && other_idx < best_idx)) {
        best_key = other_key;
        best_idx = other_idx;
      }
    }

    if ((best_idx & (METAL_SIMD_SIZE - 1)) == lane) {
      picked |= 1u << (best_idx / METAL_SIMD_SIZE);
    }

    chosen_vals[k] = top_k_score_from_key(best_key);
    chosen_ids[k] = best_idx;
  }

  if (lane == 0) {
    const float max_chosen = chosen_vals[0]; // the picks are descending
    float sum_exp = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
      const float weight = (max_chosen == -INFINITY) ? (1.0f / float(top_k)) : fast::exp(chosen_vals[k] - max_chosen);
      chosen_vals[k] = weight;
      sum_exp += weight;
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

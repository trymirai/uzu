#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/top_k.h"
#include "common.h"

template <typename T>
VARIANTS(T, float)
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
    keys[i] = top_k_score_key(float(row[lane + i * METAL_SIMD_SIZE]));
  }

  float chosen_val = 0.0f;
  uint chosen_id = 0;
  uint picked = 0;

  for (uint k = 0; k < top_k; ++k) {
    uint best_key = 0;
    uint best_idx = 0xFFFFFFFFu;
    for (uint i = 0; i < MOE_ROUTER_MAX_PER_LANE; ++i) {
      const uint expert_idx = lane + i * METAL_SIMD_SIZE;
      if ((picked & (1u << i)) == 0) {
        if (keys[i] > best_key || (keys[i] == best_key && expert_idx < best_idx)) {
          best_key = keys[i];
          best_idx = expert_idx;
        }
      }
    }

    const uint max_key = simd_max(best_key);
    best_idx = simd_min(best_key == max_key ? best_idx : 0xFFFFFFFFu);
    best_key = max_key;

    if ((best_idx & (METAL_SIMD_SIZE - 1)) == lane) {
      picked |= 1u << (best_idx / METAL_SIMD_SIZE);
    }

    if (lane == k) {
      chosen_val = top_k_score_from_key(best_key);
      chosen_id = best_idx;
    }
  }

  const float max_chosen = simd_shuffle(chosen_val, ushort(0));
  const float weight = fast::exp(chosen_val - max_chosen);
  float sum_exp = 0.0f;
  for (uint k = 0; k < top_k; ++k) {
    sum_exp += simd_shuffle(weight, ushort(k));
  }
  const float inv_sum = 1.0f / sum_exp;
  if (lane < top_k) {
    expert_ids[token_idx * top_k + lane] = chosen_id;
    expert_weights[token_idx * top_k + lane] = weight * inv_sum;
  }
}

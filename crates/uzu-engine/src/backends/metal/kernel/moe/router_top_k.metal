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
    const uint token_idx GROUPS(batch_size),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  const device T* row = logits + token_idx * logit_stride;
  float vals[MOE_ROUTER_MAX_PER_LANE];
  for (uint i = 0; i < MOE_ROUTER_MAX_PER_LANE; ++i) {
    const uint e = lane + i * METAL_SIMD_SIZE;
    if (e < num_experts) {
      float v = float(row[e]);
      if (has_biases) {
        v += float(biases[e]);
      }
      vals[i] = isnan(v) ? -INFINITY : v;
    } else {
      vals[i] = -INFINITY;
    }
  }

  float chosen_vals[MAX_TOP_K];
  uint chosen_ids[MAX_TOP_K];
  float max_chosen = -INFINITY;

  for (uint k = 0; k < top_k; ++k) {
    float best_val = -INFINITY;
    uint best_idx = 0xFFFFFFFFu;
    for (uint i = 0; i < MOE_ROUTER_MAX_PER_LANE; ++i) {
      const uint e = lane + i * METAL_SIMD_SIZE;
      if (e < num_experts) {
        const float v = vals[i];
        if (v > best_val || (v == best_val && e < best_idx)) {
          best_val = v;
          best_idx = e;
        }
      }
    }

    for (ushort offset = METAL_SIMD_SIZE / 2; offset > 0; offset >>= 1) {
      const float other_val = simd_shuffle_xor(best_val, offset);
      const uint other_idx = simd_shuffle_xor(best_idx, offset);
      if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
        best_val = other_val;
        best_idx = other_idx;
      }
    }

    if (best_idx >= num_experts) {
      best_idx = min(k, num_experts - 1);
    }

    if ((best_idx & (METAL_SIMD_SIZE - 1)) == lane) {
      const uint slot = best_idx / METAL_SIMD_SIZE;
      if (slot < MOE_ROUTER_MAX_PER_LANE) {
        vals[slot] = -INFINITY;
      }
    }

    chosen_vals[k] = best_val;
    chosen_ids[k] = best_idx;
    max_chosen = (k == 0) ? best_val : max(max_chosen, best_val);
  }

  if (lane == 0) {
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
      out_ids[k] = (chosen_ids[k] < num_experts) ? chosen_ids[k] : min(k, num_experts - 1);
      out_weights[k] = chosen_vals[k] * inv_sum;
    }
  }
}

#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "moe_commons.h"

// ============================================================================
// Pass B (fused with finalize): hidden[k] @ W2[expert] → y (directly)
// Computes: y[d] = Σ_k prob[k] * (hidden[k] @ W2[expert_k, d] + bias)
// Each simdgroup computes one final output element
// Grid: (ceil(d_model/8), 1)  - NOT per K!
// ============================================================================
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MoeExpertsDecodeSinglePassB)(
    device const float* hidden, // [K, d_ff]
    device const int* topk_ids, // [K]
    device const T* topk_probs, // [K]
    device const T* w2_all,     // [E, d_model, d_ff]
    device const T* biases,     // [E, d_model]
    device T* y,                // [d_model]
    constant uint& d_model,
    constant uint& d_ff,
    constant uint& k_input,
    const ThreadContext thread_context,
    const uint d_block GROUPS(d_model.div_ceil(8)),
    const uint tid THREADS(256)
) {
  const uint my_col = d_block * 8 + thread_context.threadgroup_index;
  if (my_col >= d_model)
    return;

  const uint vec_iters = d_ff / 128;
  const ulong w2_expert_stride = (ulong)d_model * (ulong)d_ff;

  float final_acc = 0.0f;

  // Loop over k_input experts
  for (uint k = 0; k < k_input; ++k) {
    const uint expert_u = uint(topk_ids[k]);
    const float prob = float(topk_probs[k]);

    device const float* hidden_ptr = hidden + (ulong)k * (ulong)d_ff;
    device const T* w2_ptr = w2_all + (ulong)expert_u * w2_expert_stride +
                             (ulong)my_col * (ulong)d_ff;

    float acc = 0.0f;

    // Vectorized reduction
    for (uint i = 0; i < vec_iters; ++i) {
      uint base_idx = i * 128 + thread_context.simdgroup_index * 4;

      float4 h_vec =
          *reinterpret_cast<device const float4*>(hidden_ptr + base_idx);
      device const T* w_vec2 =
          reinterpret_cast<device const T*>(w2_ptr + base_idx);
      acc += h_vec.x * float(w_vec2[0]);
      acc += h_vec.y * float(w_vec2[1]);
      acc += h_vec.z * float(w_vec2[2]);
      acc += h_vec.w * float(w_vec2[3]);
    }

    // Remainder
    for (uint idx = vec_iters * 128 + thread_context.simdgroup_index;
         idx < d_ff;
         idx += 32) {
      acc += hidden_ptr[idx] * float(w2_ptr[idx]);
    }

    // Simdgroup reduction and accumulate
    float result = simd_sum(acc);
    if (thread_context.simdgroup_index == 0) {
      result += float(biases[(ulong)expert_u * (ulong)d_model + (ulong)my_col]);
      final_acc += prob * result;
    }
  }

  if (thread_context.simdgroup_index == 0) {
    y[my_col] = T(final_acc);
  }
}
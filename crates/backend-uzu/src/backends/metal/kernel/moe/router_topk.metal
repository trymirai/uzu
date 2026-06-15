#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
using namespace metal;

constant uint THREADS_PER_TG = 256; // 8 simdgroups
constant uint MAX_EXPERTS = 512;
constant uint MAX_TOPK = 128;
constant float NEG_INF = -INFINITY;

template <typename ScalarT>
VARIANTS(ScalarT, half, bfloat, float)
PUBLIC KERNEL(MoeRouterTopK)(
    const device ScalarT* input,
    const device ScalarT* weight,
    const device ScalarT* bias OPTIONAL(has_biases),
    const device ScalarT* router_scale OPTIONAL(has_router_scales),
    const device ScalarT* per_expert_scale OPTIONAL(has_per_expert_scales),
    device int* topk_ids,
    device ScalarT* topk_probs,
    constant uint& t,
    constant uint& d_model,
    constant uint& e,
    constant uint& k,
    constant bool& renorm,
    constant float& router_norm_epsilon OPTIONAL(normalize_router_input),
    constant float& router_input_scale OPTIONAL(has_router_input_scale),
    const bool has_biases SPECIALIZE,
    const bool has_router_scales SPECIALIZE,
    const bool has_per_expert_scales SPECIALIZE,
    const bool has_router_input_scale SPECIALIZE,
    const bool normalize_router_input SPECIALIZE,
    threadgroup float4 x4_cache[1024],
    threadgroup float expert_logits[MAX_EXPERTS],
    threadgroup float reduce_tmp[THREADS_PER_TG],
    threadgroup uint reduce_tmp_u[THREADS_PER_TG],
    threadgroup uint shared_best_idx[1],
    threadgroup float shared_best_val[1],
    const ThreadContext thread_context,
    const uint tgpig_x GROUPS(1),
    const uint token_idx GROUPS(t),
    const uint lid THREADS(256)
) {
  if (d_model == 0 || e == 0 || k == 0) {
    return;
  }

  const uint vec4_count = d_model / 4u;
  const device ScalarT* token_input = input + (ulong)token_idx * (ulong)d_model;

  float local_sum_sq = 0.0f;
  for (uint vec4_idx = lid; vec4_idx < vec4_count; vec4_idx += THREADS_PER_TG) {
    const uint scalar_base = vec4_idx * 4;
    const float4 x4 = float4(
        token_input[scalar_base + 0],
        token_input[scalar_base + 1],
        token_input[scalar_base + 2],
        token_input[scalar_base + 3]
    );
    x4_cache[vec4_idx] = x4;
    if (normalize_router_input) {
      local_sum_sq += dot(x4, x4);
    }
  }

  float inv_rms = 1.0f;
  if (normalize_router_input) {
    const float sum_sq =
        threadgroup_cooperative_reduce<SimdReduceSum<float>, THREADS_PER_TG>(local_sum_sq, reduce_tmp, thread_context);
    inv_rms = rsqrt(sum_sq / float(d_model) + router_norm_epsilon);
  }

  if (normalize_router_input || has_router_input_scale || has_router_scales) {
    float4 base_scale4 = float4(inv_rms);
    if (has_router_input_scale) {
      base_scale4 *= float4(router_input_scale);
    }
    for (uint vec4_idx = lid; vec4_idx < vec4_count; vec4_idx += THREADS_PER_TG) {
      const uint scalar_base = vec4_idx * 4;
      float4 scale4 = base_scale4;
      if (has_router_scales) {
        scale4 *= float4(
            router_scale[scalar_base + 0],
            router_scale[scalar_base + 1],
            router_scale[scalar_base + 2],
            router_scale[scalar_base + 3]
        );
      }
      x4_cache[vec4_idx] *= scale4;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint expert = thread_context.simdgroup_index; expert < e; expert += thread_context.simdgroups_per_threadgroup) {
    const device ScalarT* expert_weights = weight + (ulong)expert * (ulong)d_model;

    float4 dot4 = float4(0.0f);
    for (uint vec4_idx = thread_context.simd_lane_id; vec4_idx < vec4_count; vec4_idx += 32u) {
      const uint scalar_base = vec4_idx * 4;
      const float4 w4 = float4(
          expert_weights[scalar_base + 0],
          expert_weights[scalar_base + 1],
          expert_weights[scalar_base + 2],
          expert_weights[scalar_base + 3]
      );
      const float4 x4 = x4_cache[vec4_idx];
      dot4 = fma(w4, x4, dot4);
    }
    float sum = (dot4.x + dot4.y) + (dot4.z + dot4.w);
    sum = simd_sum(sum);
    if (simd_is_first()) {
      if (has_biases) {
        sum += float(bias[expert]);
      }
      expert_logits[expert] = sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint effective_k = min(k, MAX_TOPK);
  for (uint sel = 0; sel < effective_k; ++sel) {
    float local_best = NEG_INF;
    uint local_idx = 0xFFFFFFFFu;
    for (uint expert = lid; expert < e; expert += THREADS_PER_TG) {
      float candidate = expert_logits[expert];
      if (candidate > local_best || (candidate == local_best && expert < local_idx)) {
        local_best = candidate;
        local_idx = expert;
      }
    }

    float max_val =
        threadgroup_cooperative_reduce<SimdReduceMax<float>, THREADS_PER_TG>(local_best, reduce_tmp, thread_context);

    uint candidate_id = (local_best == max_val) ? local_idx : 0xFFFFFFFFu;
    uint best_idx =
        threadgroup_cooperative_reduce<SimdReduceMin<uint>, THREADS_PER_TG>(candidate_id, reduce_tmp_u, thread_context);

    if (lid == 0) {
      shared_best_idx[0] = best_idx;
      shared_best_val[0] = max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint winner_idx = shared_best_idx[0];
    float winner_val = shared_best_val[0];
    if (lid == 0 && winner_idx < MAX_EXPERTS) {
      if (!renorm && has_per_expert_scales) {
        winner_val *= float(per_expert_scale[winner_idx]);
      }
      expert_logits[winner_idx] = NEG_INF;
      topk_ids[token_idx * k + sel] = int(winner_idx);
      topk_probs[token_idx * k + sel] = static_cast<ScalarT>(winner_val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0 && renorm && effective_k > 0) {
    device ScalarT* out_probs = topk_probs + token_idx * k;
    float max_logit = -INFINITY;
    for (uint i = 0; i < effective_k; ++i) {
      max_logit = fmax(max_logit, float(out_probs[i]));
    }
    float sum_exp = 0.0f;
    for (uint i = 0; i < effective_k; ++i) {
      sum_exp += exp(float(out_probs[i]) - max_logit);
    }
    float default_prob = 1.0f / float(effective_k);
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : default_prob;
    for (uint i = 0; i < effective_k; ++i) {
      float prob = (sum_exp > 0.0f) ? exp(float(out_probs[i]) - max_logit) * inv_sum : default_prob;
      const int expert_id = topk_ids[token_idx * k + i];
      const float expert_scale = (has_per_expert_scales && expert_id >= 0) ? float(per_expert_scale[expert_id]) : 1.0f;
      out_probs[i] = static_cast<ScalarT>(prob * expert_scale);
    }
    for (uint i = effective_k; i < k; ++i) {
      out_probs[i] = static_cast<ScalarT>(0.0f);
    }
  }
}

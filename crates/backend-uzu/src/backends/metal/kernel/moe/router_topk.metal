#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../generated/moe.h"
using namespace metal;
using namespace uzu::moe;

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
    threadgroup float4 x_cache[ROUTER_MAX_MODEL_DIM / 4],
    threadgroup float logits_shared[ROUTER_MAX_EXPERTS],
    threadgroup float reduce_tmp[ROUTER_THREADS_PER_THREADGROUP],
    threadgroup uint reduce_tmp_u[ROUTER_THREADS_PER_THREADGROUP],
    threadgroup uint shared_best_idx[1],
    threadgroup float shared_best_val[1],
    const ThreadContext thread_context,
    const uint tgpig_x GROUPS(1),
    const uint token_idx GROUPS(t),
    // The DSL launch annotation must remain a literal; the shader-side capacity is shared via gpu_types/moe.
    const uint lid THREADS(256)
) {
  if (d_model == 0 || e == 0 || k == 0) {
    return;
  }

  const uint vecs = d_model / 4u;
  const device ScalarT* x_vec = input + (ulong)token_idx * (ulong)vecs * 4;

  float local_sum_sq = 0.0f;
  for (uint c = lid; c < vecs; c += ROUTER_THREADS_PER_THREADGROUP) {
    const float4 xv = float4(x_vec[c * 4 + 0], x_vec[c * 4 + 1], x_vec[c * 4 + 2], x_vec[c * 4 + 3]);
    x_cache[c] = xv;
    if (normalize_router_input) {
      local_sum_sq += dot(xv, xv);
    }
  }

  float inv_rms = 1.0f;
  if (normalize_router_input) {
    const float sum_sq = threadgroup_cooperative_reduce<SimdReduceSum<float>, ROUTER_THREADS_PER_THREADGROUP>(
        local_sum_sq,
        reduce_tmp,
        thread_context
    );
    inv_rms = rsqrt(sum_sq / float(d_model) + router_norm_epsilon);
  }

  if (normalize_router_input || has_router_input_scale || has_router_scales) {
    float4 base_scale4 = float4(inv_rms);
    if (has_router_input_scale) {
      base_scale4 *= float4(router_input_scale);
    }
    for (uint c = lid; c < vecs; c += ROUTER_THREADS_PER_THREADGROUP) {
      float4 scale4 = base_scale4;
      if (has_router_scales) {
        scale4 *=
            float4(router_scale[c * 4 + 0], router_scale[c * 4 + 1], router_scale[c * 4 + 2], router_scale[c * 4 + 3]);
      }
      x_cache[c] *= scale4;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint row = thread_context.simdgroup_index; row < e; row += thread_context.simdgroups_per_threadgroup) {
    const device ScalarT* w_vec = weight + (ulong)row * (ulong)vecs * 4;

    float4 accum4 = float4(0.0f);
    for (uint c = thread_context.simd_lane_id; c < vecs; c += 32u) {
      const float4 wv = float4(w_vec[c * 4 + 0], w_vec[c * 4 + 1], w_vec[c * 4 + 2], w_vec[c * 4 + 3]);
      const float4 xv = x_cache[c];
      accum4 = fma(wv, xv, accum4);
    }
    float sum = (accum4.x + accum4.y) + (accum4.z + accum4.w);
    sum = simd_sum(sum);
    if (simd_is_first()) {
      if (has_biases) {
        sum += float(bias[row]);
      }
      logits_shared[row] = sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Sanitize nonfinite logits so the selection below always sees defined
  // values; a NaN can never win a comparison and poison a slot silently.
  for (uint row = lid; row < e; row += ROUTER_THREADS_PER_THREADGROUP) {
    if (!isfinite(logits_shared[row])) {
      logits_shared[row] = NEG_INF;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint effective_k = min(k, ROUTER_MAX_SELECTED_EXPERTS);
  for (uint sel = 0; sel < effective_k; ++sel) {
    float local_best = NEG_INF;
    uint local_idx = 0xFFFFFFFFu;
    for (uint row = lid; row < e; row += ROUTER_THREADS_PER_THREADGROUP) {
      float candidate = logits_shared[row];
      if (candidate > local_best || (candidate == local_best && row < local_idx)) {
        local_best = candidate;
        local_idx = row;
      }
    }

    float max_val = threadgroup_cooperative_reduce<SimdReduceMax<float>, ROUTER_THREADS_PER_THREADGROUP>(
        local_best,
        reduce_tmp,
        thread_context
    );

    uint candidate_id = (local_best == max_val) ? local_idx : 0xFFFFFFFFu;
    uint best_idx = threadgroup_cooperative_reduce<SimdReduceMin<uint>, ROUTER_THREADS_PER_THREADGROUP>(
        candidate_id,
        reduce_tmp_u,
        thread_context
    );

    if (lid == 0) {
      shared_best_idx[0] = best_idx;
      shared_best_val[0] = max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint winner_idx = shared_best_idx[0];
    float winner_val = shared_best_val[0];
    // Every slot is written unconditionally: a missing or nonfinite winner
    // yields id -1 and weight 0 instead of leaking stale buffer contents.
    if (lid == 0) {
      const bool winner_valid = winner_idx < ROUTER_MAX_EXPERTS && isfinite(winner_val);
      if (winner_idx < ROUTER_MAX_EXPERTS) {
        logits_shared[winner_idx] = NEG_INF;
      }
      if (winner_valid) {
        if (!renorm && has_per_expert_scales) {
          winner_val *= float(per_expert_scale[winner_idx]);
        }
        topk_ids[token_idx * k + sel] = int(winner_idx);
        topk_probs[token_idx * k + sel] = static_cast<ScalarT>(winner_val);
      } else {
        topk_ids[token_idx * k + sel] = -1;
        topk_probs[token_idx * k + sel] = static_cast<ScalarT>(0.0f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0 && renorm && effective_k > 0) {
    device ScalarT* out_probs = topk_probs + token_idx * k;
    const device int* out_ids = topk_ids + token_idx * k;
    // Invalid slots (id -1, prob 0) stay at zero weight and are excluded
    // from the softmax normalization.
    float max_logit = -INFINITY;
    for (uint i = 0; i < effective_k; ++i) {
      if (out_ids[i] >= 0) {
        max_logit = fmax(max_logit, float(out_probs[i]));
      }
    }
    float sum_exp = 0.0f;
    for (uint i = 0; i < effective_k; ++i) {
      if (out_ids[i] >= 0) {
        sum_exp += exp(float(out_probs[i]) - max_logit);
      }
    }
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (uint i = 0; i < effective_k; ++i) {
      const int expert_id = out_ids[i];
      if (expert_id < 0) {
        continue;
      }
      const float prob = exp(float(out_probs[i]) - max_logit) * inv_sum;
      const float expert_scale = has_per_expert_scales ? float(per_expert_scale[expert_id]) : 1.0f;
      out_probs[i] = static_cast<ScalarT>(prob * expert_scale);
    }
    for (uint i = effective_k; i < k; ++i) {
      topk_ids[token_idx * k + i] = -1;
      out_probs[i] = static_cast<ScalarT>(0.0f);
    }
  }
}

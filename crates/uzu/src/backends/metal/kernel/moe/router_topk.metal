#include <metal_stdlib>
#include <metal_simdgroup>
#include "../definitions.metal"
using namespace metal;

constant uint THREADS_PER_TG = 256; // 8 simdgroups
constant uint MAX_EXPERTS = 512;
constant uint MAX_TOPK = 128;
constant float NEG_INF = -INFINITY;

template <typename ScalarT>
VARIANTS(ScalarT, half, bfloat, float)
KERNEL(MoeRouterTopK)(
    const device ScalarT* input,
    const device ScalarT* weight,
    const device ScalarT* bias,
    device int* topk_ids,
    device ScalarT* topk_probs,
    constant uint& t,
    constant uint& d_model,
    constant uint& e,
    constant uint& k,
    constant bool& renorm,
    threadgroup float4 x_cache[1024],
    threadgroup float logits_shared[MAX_EXPERTS],
    threadgroup uint idx_shared[MAX_EXPERTS],
    threadgroup float reduce_tmp[THREADS_PER_TG],
    threadgroup uint reduce_tmp_u[THREADS_PER_TG],
    threadgroup uint shared_best_idx[1],
    threadgroup float shared_best_val[1],
    const Simd simd,
    const uint tgpig_x GROUPS(1),
    const uint token_idx GROUPS(t),
    const uint lid THREADS(256)
) {
  if (d_model == 0 || e == 0 || k == 0) {
    return;
  }

  const uint vecs = d_model / 4u;
  const device ScalarT* x_vec = input + (ulong)token_idx * (ulong)vecs * 4;

  for (uint c = lid; c < vecs; c += THREADS_PER_TG) {
    x_cache[c] = float4(
        x_vec[c * 4 + 0],
        x_vec[c * 4 + 1],
        x_vec[c * 4 + 2],
        x_vec[c * 4 + 3]
    );
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint row = simd.group_idx; row < e; row += simd.groups_per_threadgroup) {
    const device ScalarT* w_vec = weight + (ulong)row * (ulong)vecs * 4;

    float4 accum4 = float4(0.0f);
    for (uint c = simd.lane_idx; c < vecs; c += 32u) {
      const float4 wv = float4(
          w_vec[c * 4 + 0],
          w_vec[c * 4 + 1],
          w_vec[c * 4 + 2],
          w_vec[c * 4 + 3]
      );
      const float4 xv = x_cache[c];
      accum4 = fma(wv, xv, accum4);
    }
    float sum = (accum4.x + accum4.y) + (accum4.z + accum4.w);
    sum = simd_sum(sum);
    if (simd_is_first()) {
      sum += float(bias[row]);
      logits_shared[row] = sum;
      idx_shared[row] = row;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint row = lid + e; row < MAX_EXPERTS; row += THREADS_PER_TG) {
    logits_shared[row] = NEG_INF;
    idx_shared[row] = row;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint effective_k = min(k, MAX_TOPK);
  for (uint sel = 0; sel < effective_k; ++sel) {
    float local_best = NEG_INF;
    uint local_idx = 0xFFFFFFFFu;
    for (uint row = lid; row < e; row += THREADS_PER_TG) {
      float candidate = logits_shared[row];
      if (candidate > local_best ||
          (candidate == local_best && row < local_idx)) {
        local_best = candidate;
        local_idx = row;
      }
    }

    float max_val = threadgroup_cooperative_reduce_max<THREADS_PER_TG>(
        local_best,
        reduce_tmp,
        lid
    );

    uint candidate_id = (local_best == max_val) ? local_idx : 0xFFFFFFFFu;
    uint best_idx = threadgroup_cooperative_reduce_min<THREADS_PER_TG>(
        candidate_id,
        reduce_tmp_u,
        lid
    );

    if (lid == 0) {
      shared_best_idx[0] = best_idx;
      shared_best_val[0] = max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint winner_idx = shared_best_idx[0];
    float winner_val = shared_best_val[0];
    if (winner_idx < MAX_EXPERTS) {
      if (lid == 0) {
        logits_shared[winner_idx] = NEG_INF;
        idx_shared[winner_idx] = 0xFFFFFFFFu;
        device int* out_ids = topk_ids + token_idx * k;
        device ScalarT* out_probs = topk_probs + token_idx * k;
        out_ids[sel] = int(winner_idx);
        out_probs[sel] = static_cast<ScalarT>(winner_val);
      }
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
      float prob = (sum_exp > 0.0f)
                       ? exp(float(out_probs[i]) - max_logit) * inv_sum
                       : default_prob;
      out_probs[i] = static_cast<ScalarT>(prob);
    }
    for (uint i = effective_k; i < k; ++i) {
      out_probs[i] = static_cast<ScalarT>(0.0f);
    }
  }
}
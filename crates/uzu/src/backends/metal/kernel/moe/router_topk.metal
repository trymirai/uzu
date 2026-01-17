#include <metal_stdlib>
#include <metal_simdgroup>
#include "../definitions.metal"
using namespace metal;

constant uint THREADS_PER_TG = 256; // 8 simdgroups
constant uint MAX_EXPERTS = 512;
constant uint MAX_TOPK = 128;
constant float NEG_INF = -INFINITY;

template <typename Vec4T, typename ScalarT, typename ProbT>
inline void moe_router_topk_impl(
    const device Vec4T* input,  // [T, d_model/4]
    const device Vec4T* weight, // [E, d_model/4]
    const device ScalarT* bias, // [E]
    device int* topk_ids,       // [T, K]
    device ProbT* topk_probs,   // [T, K]
    uint T,
    uint d_model,
    uint E,
    uint K,
    uint renorm,
    uint2 tgpig,
    ushort lid,
    ushort simd_lane,
    ushort simdgroup_idx,
    ushort simdgroups_per_tg,
    threadgroup float4* x_cache,
    threadgroup float* logits_shared,
    threadgroup uint* idx_shared,
    threadgroup float* reduce_tmp,
    threadgroup uint* reduce_tmp_u,
    threadgroup uint* shared_best_idx,
    threadgroup float* shared_best_val
) {
  const uint token_idx = tgpig.y;
  if (token_idx >= T || d_model == 0 || E == 0 || K == 0) {
    return;
  }

  const uint vecs = d_model / 4u;
  const device Vec4T* x_vec = input + (ulong)token_idx * (ulong)vecs;

  for (uint c = lid; c < vecs; c += THREADS_PER_TG) {
    x_cache[c] = float4(x_vec[c]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint row = simdgroup_idx; row < E; row += simdgroups_per_tg) {
    const device Vec4T* w_vec = weight + (ulong)row * (ulong)vecs;

    float4 accum4 = float4(0.0f);
    for (uint c = simd_lane; c < vecs; c += 32u) {
      const float4 wv = float4(w_vec[c]);
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

  for (uint row = lid + E; row < MAX_EXPERTS; row += THREADS_PER_TG) {
    logits_shared[row] = NEG_INF;
    idx_shared[row] = row;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint effective_k = min(K, MAX_TOPK);
  for (uint sel = 0; sel < effective_k; ++sel) {
    float local_best = NEG_INF;
    uint local_idx = 0xFFFFFFFFu;
    for (uint row = lid; row < E; row += THREADS_PER_TG) {
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
      *shared_best_idx = best_idx;
      *shared_best_val = max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint winner_idx = *shared_best_idx;
    float winner_val = *shared_best_val;
    if (winner_idx < MAX_EXPERTS) {
      if (lid == 0) {
        logits_shared[winner_idx] = NEG_INF;
        idx_shared[winner_idx] = 0xFFFFFFFFu;
        device int* out_ids = topk_ids + token_idx * K;
        device ProbT* out_probs = topk_probs + token_idx * K;
        out_ids[sel] = int(winner_idx);
        out_probs[sel] = ProbT(winner_val);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0 && renorm != 0 && effective_k > 0) {
    device ProbT* out_probs = topk_probs + token_idx * K;
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
      out_probs[i] = ProbT(prob);
    }
    for (uint i = effective_k; i < K; ++i) {
      out_probs[i] = ProbT(0.0f);
    }
  }
}

#define DEFINE_ROUTER_TOPK_KERNEL(SUFFIX, VEC4, SCALAR, PROB)                  \
  [[max_total_threads_per_threadgroup(256)]]                                   \
  kernel void moe_router_topk_##SUFFIX(                                        \
      const device VEC4* input [[buffer(0)]],                                  \
      const device VEC4* weight [[buffer(1)]],                                 \
      const device SCALAR* bias [[buffer(2)]],                                 \
      device int* topk_ids [[buffer(3)]],                                      \
      device PROB* topk_probs [[buffer(4)]],                                   \
      constant uint& T [[buffer(5)]],                                          \
      constant uint& d_model [[buffer(6)]],                                    \
      constant uint& E [[buffer(7)]],                                          \
      constant uint& K [[buffer(8)]],                                          \
      constant uint& renorm [[buffer(9)]],                                     \
      uint2 tgpig [[threadgroup_position_in_grid]],                            \
      ushort lid [[thread_index_in_threadgroup]],                              \
      ushort simd_lane [[thread_index_in_simdgroup]],                          \
      ushort simdgroup_idx [[simdgroup_index_in_threadgroup]],                 \
      ushort simdgroups_per_tg [[simdgroups_per_threadgroup]]                  \
  ) {                                                                          \
    threadgroup float4 x_cache[1024];                                          \
    threadgroup float logits_shared[MAX_EXPERTS];                              \
    threadgroup uint idx_shared[MAX_EXPERTS];                                  \
    threadgroup float reduce_tmp[THREADS_PER_TG];                              \
    threadgroup uint reduce_tmp_u[THREADS_PER_TG];                             \
    threadgroup uint shared_best_idx_mem;                                      \
    threadgroup float shared_best_val_mem;                                     \
    moe_router_topk_impl<VEC4, SCALAR, PROB>(                                  \
        input,                                                                 \
        weight,                                                                \
        bias,                                                                  \
        topk_ids,                                                              \
        topk_probs,                                                            \
        T,                                                                     \
        d_model,                                                               \
        E,                                                                     \
        K,                                                                     \
        renorm,                                                                \
        tgpig,                                                                 \
        lid,                                                                   \
        simd_lane,                                                             \
        simdgroup_idx,                                                         \
        simdgroups_per_tg,                                                     \
        x_cache,                                                               \
        logits_shared,                                                         \
        idx_shared,                                                            \
        reduce_tmp,                                                            \
        reduce_tmp_u,                                                          \
        &shared_best_idx_mem,                                                  \
        &shared_best_val_mem                                                   \
    );                                                                         \
  }

DEFINE_ROUTER_TOPK_KERNEL(f16, half4, half, half)
DEFINE_ROUTER_TOPK_KERNEL(bf16, bfloat4, bfloat, bfloat)
DEFINE_ROUTER_TOPK_KERNEL(f32, float4, float, float)

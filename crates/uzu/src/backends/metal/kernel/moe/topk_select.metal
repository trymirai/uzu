#include <metal_stdlib>
#include "../definitions.metal"
using namespace metal;

constant uint THREADS_PER_TG = 128;
constant uint MAX_EXPERTS = 512;
constant uint MAX_TOPK = 128;
constant float NEG_INF = -INFINITY;

template<typename LogitT, typename ProbT>
inline void moe_topk_select_impl(
    device const LogitT* logits,
    device int* topk_ids,
    device ProbT* topk_probs,
    uint T,
    uint E,
    uint K,
    uint renorm,
    ushort lid,
    uint token,
    threadgroup float* tg_logits,
    threadgroup uint* tg_indices,
    threadgroup float* reduce_tmp,
    threadgroup uint* reduce_tmp_u,
    threadgroup float* selected_vals,
    threadgroup uint* selected_ids)
{
    if (token >= T || K == 0) {
        return;
    }

    const uint effective_k = min(K, MAX_TOPK);

    for (uint i = lid; i < MAX_EXPERTS; i += THREADS_PER_TG) {
        if (i < E) {
            tg_logits[i] = float(logits[token * E + i]);
        } else {
            tg_logits[i] = NEG_INF;
        }
        tg_indices[i] = i;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint selections = min(effective_k, E);
    for (uint sel = 0; sel < selections; ++sel) {
        float local_best_val = NEG_INF;
        uint local_best_idx = 0xFFFFFFFFu;

        for (uint i = lid; i < E; i += THREADS_PER_TG) {
            float candidate = tg_logits[i];
            if (candidate > local_best_val ||
                (candidate == local_best_val && i < local_best_idx)) {
                local_best_val = candidate;
                local_best_idx = i;
            }
        }

        float max_val = threadgroup_cooperative_reduce_max<THREADS_PER_TG>(
            local_best_val,
            reduce_tmp,
            lid);

        uint candidate_id =
            (local_best_val == max_val) ? local_best_idx : 0xFFFFFFFFu;
        uint best_idx = threadgroup_cooperative_reduce_min<THREADS_PER_TG>(
            candidate_id,
            reduce_tmp_u,
            lid);

        if (lid == 0) {
            selected_vals[sel] = max_val;
            selected_ids[sel] = best_idx;
            tg_logits[best_idx] = NEG_INF;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        device int* out_ids = topk_ids + token * K;
        device ProbT* out_probs = topk_probs + token * K;

        const uint written = selections;
        for (uint i = 0; i < written; ++i) {
            out_ids[i] = int(selected_ids[i]);
        }
        for (uint i = written; i < K; ++i) {
            out_ids[i] = 0;
        }

        if (renorm != 0 && written > 0) {
            float max_logit = -INFINITY;
            for (uint i = 0; i < written; ++i) {
                max_logit = fmax(max_logit, selected_vals[i]);
            }

            float exp_values[MAX_TOPK];
            float sum_exp = 0.0f;
            for (uint i = 0; i < written; ++i) {
                float e_val = exp(selected_vals[i] - max_logit);
                exp_values[i] = e_val;
                sum_exp += e_val;
            }

            float default_prob = 1.0f / float(written);
            float inv_sum =
                (sum_exp > 0.0f) ? (1.0f / sum_exp) : default_prob;

            for (uint i = 0; i < written; ++i) {
                float prob = (sum_exp > 0.0f) ? (exp_values[i] * inv_sum)
                                              : default_prob;
                out_probs[i] = ProbT(prob);
            }
            for (uint i = written; i < K; ++i) {
                out_probs[i] = ProbT(0.0f);
            }
        } else {
            for (uint i = 0; i < written; ++i) {
                out_probs[i] = ProbT(selected_vals[i]);
            }
            for (uint i = written; i < K; ++i) {
                out_probs[i] = ProbT(0.0f);
            }
        }
    }
}

#define DEFINE_MOE_TOPK_KERNEL(LOGIT_T, PROB_T, SUFFIX) \
kernel void moe_topk_select_##SUFFIX( \
    device const LOGIT_T* logits [[buffer(0)]], \
    device int* topk_ids [[buffer(1)]], \
    device PROB_T* topk_probs [[buffer(2)]], \
    constant uint& T [[buffer(3)]], \
    constant uint& E [[buffer(4)]], \
    constant uint& K [[buffer(5)]], \
    constant uint& renorm [[buffer(6)]], \
    ushort lid [[thread_index_in_threadgroup]], \
    uint3 tgpig [[threadgroup_position_in_grid]]) \
{ \
    threadgroup float tg_logits[MAX_EXPERTS]; \
    threadgroup uint tg_indices[MAX_EXPERTS]; \
    threadgroup float reduce_tmp[THREADS_PER_TG]; \
    threadgroup uint reduce_tmp_u[THREADS_PER_TG]; \
    threadgroup float selected_vals[MAX_TOPK]; \
    threadgroup uint selected_ids[MAX_TOPK]; \
    moe_topk_select_impl<LOGIT_T, PROB_T>( \
        logits, topk_ids, topk_probs, T, E, K, renorm, lid, tgpig.x, \
        tg_logits, tg_indices, reduce_tmp, reduce_tmp_u, selected_vals, selected_ids); \
}

DEFINE_MOE_TOPK_KERNEL(half, half, f16)
DEFINE_MOE_TOPK_KERNEL(float, half, f32)
DEFINE_MOE_TOPK_KERNEL(bfloat, bfloat, bf16)

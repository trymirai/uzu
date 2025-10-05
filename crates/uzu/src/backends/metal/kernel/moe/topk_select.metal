#include <metal_stdlib>
using namespace metal;

// Kernel expects:
// buffer(0): logits [T * E] as half or float (two entrypoints below)
// buffer(1): topk_ids [T * K] as int32
// buffer(2): topk_probs [T * K] as half
// bytes(3): T (u32)
// bytes(4): E (u32)
// bytes(5): K (u32)
// bytes(6): renorm (u32: 0/1)

template <typename LogitT, typename ProbT>
inline void topk_select_impl(
    device const LogitT* logits,
    device int* topk_ids,
    device ProbT* topk_probs,
    uint T,
    uint E,
    uint K,
    uint renorm,
    uint gid)
{
    if (gid >= T) return;
    const uint row_off = gid * E;

    float best_vals[4];
    int   best_ids[4];
    for (uint i = 0; i < 4; ++i) { best_vals[i] = -INFINITY; best_ids[i] = -1; }

    // linear scan over experts
    for (uint e = 0; e < E; ++e) {
        float v = float(logits[row_off + e]);
        // insert into descending best_vals with deterministic tie-break
        int insert_pos = -1;
        uint Kc_u = (K < 4u ? K : 4u);
        for (int j = int(Kc_u) - 1; j >= 0; --j) {
            if (v > best_vals[j] || (v == best_vals[j] && int(e) < best_ids[j])) {
                insert_pos = j;
            }
        }
        if (insert_pos >= 0) {
            // shift down
            for (int s = int(Kc_u) - 1; s > insert_pos; --s) {
                best_vals[s] = best_vals[s - 1];
                best_ids[s] = best_ids[s - 1];
            }
            best_vals[insert_pos] = v;
            best_ids[insert_pos] = int(e);
        }
    }

    // write ids
    const uint out_off = gid * K;
    for (uint k = 0; k < K; ++k) {
        topk_ids[out_off + k] = best_ids[k];
    }

    // probs: either logits or renormalized softmax over selected K
    if (renorm != 0) {
        // stabilize with max for numerical safety
        float max_v = -INFINITY;
        for (uint k = 0; k < K; ++k) { max_v = fmax(max_v, best_vals[k]); }
        float sum = 0.0f;
        float expv[4];
        for (uint k = 0; k < K; ++k) {
            expv[k] = exp(best_vals[k] - max_v);
            sum += expv[k];
        }
        for (uint k = 0; k < K; ++k) {
            float p = (sum > 0.0f) ? (expv[k] / sum) : (1.0f / float(K));
            topk_probs[out_off + k] = ProbT(p);
        }
    } else {
        for (uint k = 0; k < K; ++k) {
            topk_probs[out_off + k] = ProbT(best_vals[k]);
        }
    }
}

kernel void moe_topk_select_f16(
    device const half* logits [[buffer(0)]],
    device int* topk_ids [[buffer(1)]],
    device half* topk_probs [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& E [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& renorm [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    topk_select_impl<half, half>(logits, topk_ids, topk_probs, T, E, K, renorm, gid);
}

kernel void moe_topk_select_f32(
    device const float* logits [[buffer(0)]],
    device int* topk_ids [[buffer(1)]],
    device half* topk_probs [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& E [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& renorm [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    topk_select_impl<float, half>(logits, topk_ids, topk_probs, T, E, K, renorm, gid);
}

kernel void moe_topk_select_bf16(
    device const bfloat* logits [[buffer(0)]],
    device int* topk_ids [[buffer(1)]],
    device bfloat* topk_probs [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& E [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& renorm [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    topk_select_impl<bfloat, bfloat>(logits, topk_ids, topk_probs, T, E, K, renorm, gid);
}


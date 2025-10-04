#include <metal_stdlib>
using namespace metal;

// Defaults must match Rust launcher
#define BM 32
#define BN 64

kernel void moe_finalize_f16(
    device const int*  tok2row   [[buffer(0)]],  // [T*K], -1 if dropped
    device const half* probs     [[buffer(1)]],  // [T*K]
    device const half* Y_partial [[buffer(2)]],  // [sum_k, d_model]
    device half*       Y         [[buffer(3)]],  // [T, d_model]
    constant uint& T            [[buffer(4)]],
    constant uint& d_model      [[buffer(5)]],
    constant uint& K            [[buffer(6)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    if (T == 0u || d_model == 0u) return;
    const uint tile_m0 = tgpig.y * BM;
    const uint tile_n0 = tgpig.x * BN;
    if (tile_m0 >= T || tile_n0 >= d_model) return;
    const uint m_rows = min((uint)BM, T - tile_m0);
    const uint n_cols = min((uint)BN, d_model - tile_n0);

    // 128 threads per TG expected
    for (uint idx = lid; idx < m_rows * n_cols; idx += 128u) {
        const uint mi = idx / n_cols;
        const uint nj = idx % n_cols;
        const uint t = tile_m0 + mi;
        const uint f = tile_n0 + nj;
        float acc = 0.0f;
        const uint base = t * K;
        for (uint k = 0; k < K; ++k) {
            const int row = tok2row[base + k];
            if (row >= 0) {
                const ulong yidx = (ulong)(uint)row * (ulong)d_model + (ulong)f;
                float prob = (float)probs[base + k];
                if (!isfinite(prob)) {
                    prob = 0.0f;
                }
                float val = (float)Y_partial[yidx];
                if (!isfinite(val)) {
                    val = 0.0f;
                }
                acc = fma(prob, val, acc);
            }
        }
        if (!isfinite(acc)) {
            acc = 0.0f;
        }
        Y[t * d_model + f] = (half)acc;
    }
}

kernel void moe_finalize_bf16(
    device const int*  tok2row   [[buffer(0)]],  // [T*K], -1 if dropped
    device const bfloat* probs   [[buffer(1)]],  // [T*K]
    device const bfloat* Y_partial [[buffer(2)]],  // [sum_k, d_model]
    device bfloat*     Y         [[buffer(3)]],  // [T, d_model]
    constant uint& T            [[buffer(4)]],
    constant uint& d_model      [[buffer(5)]],
    constant uint& K            [[buffer(6)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    if (T == 0u || d_model == 0u) return;
    const uint tile_m0 = tgpig.y * BM;
    const uint tile_n0 = tgpig.x * BN;
    if (tile_m0 >= T || tile_n0 >= d_model) return;
    const uint m_rows = min((uint)BM, T - tile_m0);
    const uint n_cols = min((uint)BN, d_model - tile_n0);

    for (uint idx = lid; idx < m_rows * n_cols; idx += 128u) {
        const uint mi = idx / n_cols;
        const uint nj = idx % n_cols;
        const uint t = tile_m0 + mi;
        const uint f = tile_n0 + nj;
        float acc = 0.0f;
        const uint base = t * K;
        for (uint k = 0; k < K; ++k) {
            const int row = tok2row[base + k];
            if (row >= 0) {
                const ulong yidx = (ulong)(uint)row * (ulong)d_model + (ulong)f;
                float prob = (float)probs[base + k];
                if (!isfinite(prob)) {
                    prob = 0.0f;
                }
                float val = (float)Y_partial[yidx];
                if (!isfinite(val)) {
                    val = 0.0f;
                }
                acc = fma(prob, val, acc);
            }
        }
        if (!isfinite(acc)) {
            acc = 0.0f;
        }
        Y[t * d_model + f] = bfloat(acc);
    }
}

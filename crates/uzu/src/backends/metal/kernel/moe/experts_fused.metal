#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../definitions.metal"
#include "../quant_matmul/mma.h"
using namespace metal;

// Tiling parameters (conservative defaults; multiples of 8 and 32)
#define BM 32   // token rows per tile (M)
#define BN 64   // output columns per tile (N)
#define BK 32   // reduction step (K)

// Compile-time gating selection via function constant (0=GELU,1=SiLU,2=SwiGLU,3=GEGLU)
constant uint GATING_SEL [[function_constant(30)]];

static inline float gelu_approx(float x) {
    const float k0 = 0.7978845608f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    
    // For large |x|, GELU(x) ≈ x for x > 0 and ≈ 0 for x < 0
    // This avoids tanh overflow for extreme values
    if (x > 10.0f) return x;
    if (x < -10.0f) return 0.0f;
    
    float x3 = x * x * x;
    float inner = x + k1 * x3;
    float tanh_arg = k0 * inner;
    
    // Clamp tanh argument to avoid potential Metal tanh issues
    tanh_arg = clamp(tanh_arg, -10.0f, 10.0f);
    
    return 0.5f * x * (1.0f + tanh(tanh_arg));
}

static inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline uint lower_bound_offsets(device const uint* offsets, uint E, uint idx) {
    // find e such that offsets[e] <= idx < offsets[e+1]
    uint lo = 0u, hi = E;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        uint v = offsets[mid+1];
        if (idx < v) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    return lo;
}

// gating_code: left in signature for ABI compatibility; selection is via function constant
kernel void moe_fused_expert_mlp_f16(
    device const half* X [[buffer(0)]],                 // [T, d_model]
    device const int* bucketed_token_ids [[buffer(1)]], // [sum_k]
    device const uint* expert_offsets [[buffer(2)]],    // [E+1]
    device const half* W1_all [[buffer(3)]],            // [E*d_ff*d_model]
    device const half* W3_all [[buffer(4)]],            // [E*d_ff*d_model] or ignored
    device const half* W2_all [[buffer(5)]],            // [E*d_model*d_ff]
    device half* Y_partial [[buffer(6)]],               // [sum_k, d_model]
    constant uint& T [[buffer(7)]],
    constant uint& d_model [[buffer(8)]],
    constant uint& d_ff [[buffer(9)]],
    constant uint& E [[buffer(10)]],
    constant uint& gating_code [[buffer(11)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort simd_lid [[thread_index_in_simdgroup]])
{
    (void)gating_code; // compile-time specialized via GATING_SEL

    const uint expert_idx = tgpig.z;
    if (expert_idx >= E) return;

    // Expert segment bounds
    const uint seg_start = expert_offsets[expert_idx];
    const uint seg_end   = expert_offsets[expert_idx + 1];
    const uint seg_len   = (seg_end > seg_start) ? (seg_end - seg_start) : 0u;
    if (seg_len == 0u) return;

    // Tile origin in segment/token-major and in output columns
    const uint tile_m0 = tgpig.y * BM;
    if (tile_m0 >= seg_len) return;
    const uint tile_n0 = tgpig.x * BN;
    if (tile_n0 >= d_model) return;

    const uint m_rows = min((uint)BM, seg_len - tile_m0);
    const uint n_cols = min((uint)BN, d_model - tile_n0);

    // Threadgroup scratch
    // FC1 staging over Kx (d_model) and ff-chunk (BK)
    threadgroup half Xs[BM * BK];      // X chunk (BM x BK)
    threadgroup half W1s[BK * BK];     // W1 chunk (BKxBK) for current ff-chunk
    threadgroup half W3s[BK * BK];     // W3 chunk (BKxBK) (GLU only)
    threadgroup float Up[BM * BK];     // accumulators for FC1 (BM x BK)
    threadgroup float Vp[BM * BK];     // accumulators for GLU second branch

    // FC2 staging for W2
    threadgroup float W2sf[BK * BN];   // W2 chunk as float (BK x BN)
    threadgroup float Hs[BM * BK];     // Activated FC1 output (BM x BK)

    // Base pointers for expert weights
    const ulong W1_base = (ulong)expert_idx * (ulong)d_ff * (ulong)d_model;
    const ulong W3_base = (ulong)expert_idx * (ulong)d_ff * (ulong)d_model;
    const ulong W2_base = (ulong)expert_idx * (ulong)d_model * (ulong)d_ff;

    // Prefetch token ids for the BM rows in this tile
    threadgroup int tok[BM];
    for (uint idx = lid; idx < (uint)BM; idx += 128u) {
        const uint row_global = seg_start + tile_m0 + idx;
        tok[idx] = (idx < m_rows) ? bucketed_token_ids[row_global] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Set up MMA operator for FC2
    constexpr int WM = 2;
    constexpr int WN = 2;
    using mma_fc2_t = matmul_utils::BlockMMA<float, half, BM, BN, BK, WM, WN, false, false, BK, BN, float>;
    mma_fc2_t mma_op(simd_gid, simd_lid);

    // Loop over ff (d_ff) in BK-sized tiles
    for (uint ff0 = 0; ff0 < d_ff; ff0 += BK) {
        const uint ff_chunk = min((uint)BK, d_ff - ff0);

        // Zero-initialize FC1 accumulators
        for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
            Up[idx] = 0.0f;
            if (GATING_SEL > 1u) {
                Vp[idx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Loop over K dimension (d_model) in BK chunks to build FC1 outputs
        for (uint k0 = 0; k0 < d_model; k0 += BK) {
            const uint k_chunk = min((uint)BK, d_model - k0);

            // Cooperative load Xs (BM x k_chunk)
            for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
                const uint mi = idx / BK;
                const uint kk = idx % BK;
                half val = (half)0.0h;
                if (mi < m_rows && kk < k_chunk) {
                    const int token_id = tok[mi];
                    if (token_id >= 0) {
                        const ulong xg = (ulong)(uint)token_id * (ulong)d_model + (ulong)(k0 + kk);
                        val = X[xg];
                    }
                }
                Xs[mi * BK + kk] = val;
            }

            // Cooperative load W1s (k_chunk x ff_chunk) packed as [kk, r]
            for (uint idx = lid; idx < (uint)(BK * BK); idx += 128u) {
                const uint kk = idx / BK;
                const uint r = idx % BK;
                half val1 = (half)0.0h;
                half val3 = (half)0.0h;
                if (kk < k_chunk && r < ff_chunk) {
                    const ulong w1g = W1_base + (ulong)(ff0 + r) * (ulong)d_model + (ulong)(k0 + kk);
                    val1 = W1_all[w1g];
                    if (GATING_SEL > 1u) {
                        const ulong w3g = W3_base + (ulong)(ff0 + r) * (ulong)d_model + (ulong)(k0 + kk);
                        val3 = W3_all[w3g];
                    }
                }
                W1s[kk * BK + r] = val1;
                if (GATING_SEL > 1u) {
                    W3s[kk * BK + r] = val3;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate Up (and Vp) for this k_chunk cooperatively over (mi, r)
            for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
                const uint mi = idx / BK;
                const uint r = idx % BK;
                float s1 = 0.0f;
                float s3 = 0.0f;
                for (uint kk = 0; kk < k_chunk; ++kk) {
                    const float xv = (float)Xs[mi * BK + kk];
                    s1 += xv * (float)W1s[kk * BK + r];
                    if (GATING_SEL > 1u) {
                        s3 += xv * (float)W3s[kk * BK + r];
                    }
                }
                Up[mi * BK + r] += s1;
                if (GATING_SEL > 1u) {
                    Vp[mi * BK + r] += s3;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Apply activation/gating to produce Hs (BM x ff_chunk)
        for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
            const uint mi = idx / BK;
            const uint r = idx % BK;
            if (mi < m_rows && r < ff_chunk) {
                const float up = Up[idx];
                float h;
                if (GATING_SEL <= 1u) {
                    h = (GATING_SEL == 0u) ? gelu_approx(up) : silu(up);
                } else {
                    const float g = (GATING_SEL == 2u) ? silu(up) : gelu_approx(up);
                    h = g * Vp[idx];
                }
                Hs[idx] = h;
            } else {
                Hs[idx] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative load W2sf for this ff_chunk and our N tile (BK x BN)
        for (uint idx = lid; idx < (uint)(BK * BN); idx += 128u) {
            const uint r = idx / BN;
            const uint nj = idx % BN;
            float v = 0.0f;
            if (r < ff_chunk && nj < n_cols) {
                const ulong w2g = W2_base + (ulong)(tile_n0 + nj) * (ulong)d_ff + (ulong)(ff0 + r);
                v = (float)W2_all[w2g];
            }
            W2sf[r * BN + nj] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate MMA tile: C += Hs (BM x BK) * W2sf (BK x BN) in float
        mma_op.mma(Hs, W2sf);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Store MMA tile to Y_partial with tail handling
    const ulong y_base = (ulong)(seg_start + tile_m0) * (ulong)d_model + (ulong)tile_n0;
    device half* y_ptr = Y_partial + (uint)y_base;
    const short num_els = (short)m_rows;
    const short num_outs = (short)n_cols;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (num_els < (short)BM || num_outs < (short)BN) {
        mma_op.store_result_safe(y_ptr, (int)d_model, short2(num_outs, num_els));
    } else {
        mma_op.store_result(y_ptr, (int)d_model);
    }
}



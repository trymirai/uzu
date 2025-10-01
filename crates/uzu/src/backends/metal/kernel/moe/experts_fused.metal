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
constant uint DEBUG_MASK [[function_constant(31)]];

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

static inline float silu(float x, float alpha) {
    return x / (1.0f + exp(-alpha * x));
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

// Find expert index e such that tile_row_offsets[e] <= y < tile_row_offsets[e+1]
static inline uint expert_for_tile(device const uint* tile_row_offsets, uint E, uint y) {
    uint lo = 0u, hi = E;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        uint v = tile_row_offsets[mid + 1u];
        if (y < v) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    return lo;
}

// PERSISTENT KERNEL: loops over experts internally (like vLLM grouped GEMM)
// Dispatch as 2D grid (N_tiles, M_tiles_total) instead of 3D (N_tiles, M_tiles, E)
// This avoids allocating threadgroup memory for empty experts
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
    device const half* up_biases [[buffer(12)]],        // [E*d_ff] or [E*2*d_ff] if fused
    device const half* down_biases [[buffer(13)]],      // [E*d_model]
    constant float& gate_clip_min [[buffer(14)]],
    constant float& gate_clip_max [[buffer(15)]],
    constant float& up_clip_min [[buffer(16)]],
    constant float& up_clip_max [[buffer(17)]],
    constant float& silu_alpha [[buffer(18)]],
    device const uint* tile_row_offsets [[buffer(19)]], // [E+1]
    uint lid [[thread_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort simd_lid [[thread_index_in_simdgroup]])
{
    (void)gating_code; // compile-time specialized via GATING_SEL

    const uint tile_n0 = tgpig.x * BN;  // Column tile
    if (tile_n0 >= d_model) return;
    
    const uint n_cols = min((uint)BN, d_model - tile_n0);
    
    // Threadgroup scratch (declared once, reused for each expert iteration)
    threadgroup half Xs[BM * BK];      // X chunk (BM x BK)
    threadgroup half Wk[BK * BK];      // Weight chunk (BKxBK) for W1/W3
    threadgroup float Htile[BM * BK];  // FC1 temporary accumulators (BM x BK)
    threadgroup float W2sf[BK * BN];   // W2 chunk as float (BK x BN)
    threadgroup float Hs[BM * BK];     // Activated FC1 output (BM x BK)
    threadgroup int tok[BM];           // Token IDs for this tile
    
    // Compact tile path using tile_row_offsets
    const uint total_tiles = tile_row_offsets[E];
    const uint y_id = tgpig.y;
    if (y_id >= total_tiles) return;
    const uint expert_idx = expert_for_tile(tile_row_offsets, E, y_id);
    const uint seg_start = expert_offsets[expert_idx];
    const uint seg_end   = expert_offsets[expert_idx + 1u];
    const uint tiles_before = tile_row_offsets[expert_idx];
    const uint tile_local = y_id - tiles_before;
    const uint tile_m0 = tile_local * BM;
    const uint seg_len = (seg_end > seg_start) ? (seg_end - seg_start) : 0u;
    if (seg_len == 0u) return;
    if (tile_m0 >= seg_len) return;
    const uint m_rows = min((uint)BM, seg_len - tile_m0);

    // Base pointers for expert weights
    const ulong W1_base = (ulong)expert_idx * (ulong)d_ff * (ulong)d_model;
    const ulong W3_base = (ulong)expert_idx * (ulong)d_ff * (ulong)d_model;
    const ulong W2_base = (ulong)expert_idx * (ulong)d_model * (ulong)d_ff;

    // Prefetch token ids for the BM rows in this tile
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

        // FC1 using MMA: compute U (and V for GLU) tiles of size (BM x ff_chunk)
        // U tile accumulation
        for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
            Htile[idx] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        matmul_utils::BlockMMA<half, float, BM, BK, BK, 2, 2, false, false, BK, BK, float> mma_fc1(simd_gid, simd_lid);
        for (uint k0 = 0; k0 < d_model; k0 += BK) {
            const uint k_chunk = min((uint)BK, d_model - k0);
            // Load X tile
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
            // Load W1 panel (K x ff_chunk -> BK x BK)
            for (uint idx = lid; idx < (uint)(BK * BK); idx += 128u) {
                const uint kk = idx / BK;
                const uint r = idx % BK;
                half v = (half)0.0h;
                if (kk < k_chunk && r < ff_chunk) {
                    const ulong w1g = W1_base + (ulong)(ff0 + r) * (ulong)d_model + (ulong)(k0 + kk);
                    v = W1_all[w1g];
                }
                Wk[kk * BK + r] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_fc1.mma(Xs, Wk);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Store U tile to threadgroup Htile (float)
        mma_fc1.store_result_tg(Htile, (int)BK);

        // Add up biases (W1 biases) - first d_ff elements
        for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
            const uint mi = idx / BK;
            const uint r = idx % BK;
            if (mi < m_rows && r < ff_chunk) {
                const ulong bias_idx = (ulong)expert_idx * (ulong)(d_ff * 2) + (ulong)(ff0 + r);
                float val = Htile[idx] + (float)up_biases[bias_idx];
                Htile[idx] = clamp(val, up_clip_min, up_clip_max);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (GATING_SEL > 1u) {
            // V tile accumulation for GLU
            for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
                Hs[idx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            matmul_utils::BlockMMA<half, float, BM, BK, BK, 2, 2, false, false, BK, BK, float> mma_fc1_v(simd_gid, simd_lid);
            for (uint k0 = 0; k0 < d_model; k0 += BK) {
                const uint k_chunk = min((uint)BK, d_model - k0);
                // Load X tile
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
                // Load W3 panel (K x ff_chunk)
                for (uint idx = lid; idx < (uint)(BK * BK); idx += 128u) {
                    const uint kk = idx / BK;
                    const uint r = idx % BK;
                    half v = (half)0.0h;
                    if (kk < k_chunk && r < ff_chunk) {
                        const ulong w3g = W3_base + (ulong)(ff0 + r) * (ulong)d_model + (ulong)(k0 + kk);
                        v = W3_all[w3g];
                    }
                    Wk[kk * BK + r] = v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                mma_fc1_v.mma(Xs, Wk);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            // Store V tile to Hs temporarily
            mma_fc1_v.store_result_tg(Hs, (int)BK);
            
            // Add gate biases (W3 biases) - second d_ff elements
            for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
                const uint mi = idx / BK;
                const uint r = idx % BK;
                if (mi < m_rows && r < ff_chunk) {
                    const ulong bias_idx = (ulong)expert_idx * (ulong)(d_ff * 2) + (ulong)d_ff + (ulong)(ff0 + r);
                    float val = Hs[idx] + (float)up_biases[bias_idx];
                    Hs[idx] = metal::min(val, gate_clip_max);  // Only max clipping for gate
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Apply activation/gating to produce Hs (BM x ff_chunk)
        for (uint idx = lid; idx < (uint)(BM * BK); idx += 128u) {
            const uint mi = idx / BK;
            const uint r = idx % BK;
            if (mi < m_rows && r < ff_chunk) {
                const float up = Htile[idx];
                if (GATING_SEL <= 1u) {
                    Hs[idx] = (GATING_SEL == 0u) ? gelu_approx(up) : silu(up, silu_alpha);
                } else {
                    // GLU variant (GPT-OSS): activation(gate) * up + activation(gate)
                    const float gate = Hs[idx];
                    const float swish_y = (GATING_SEL == 2u) ? silu(gate, silu_alpha) : gelu_approx(gate);
                    Hs[idx] = swish_y * up + swish_y;
                }
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
    
    // Add down biases
    for (uint idx = lid; idx < (uint)(m_rows * n_cols); idx += 128u) {
        const uint mi = idx / n_cols;
        const uint nj = idx % n_cols;
        if (mi < m_rows && nj < n_cols) {
            const ulong bias_idx = (ulong)expert_idx * (ulong)d_model + (ulong)(tile_n0 + nj);
            const ulong y_idx = y_base + (ulong)mi * (ulong)d_model + (ulong)nj;
            Y_partial[y_idx] = (half)((float)Y_partial[y_idx] + (float)down_biases[bias_idx]);
        }
    }
}



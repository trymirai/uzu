#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "../quant_matmul/mma.h"
using namespace metal;

// ============================================================================
// 2-Pass MoE Prefill: Optimized for minimal barriers
// ============================================================================
//
// Pass A: X @ W13 → Hidden (with gating + activation)
// Pass B: Hidden @ W2 → Output
//
// Layouts:
//   X_perm: [total_rows, D]
//   W13: [E, 2*FF, D] 
//   Hidden: [total_rows, FF]
//   W2: [E, FF, D]
//   Output: [total_rows, D]
//

constant uint GATING_SEL [[function_constant(30)]]; // 0=GELU, 1=SiLU, 2=SwiGLU, 3=GEGLU

static inline float gelu_approx(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    if (x > 10.0f) return x;
    if (x < -10.0f) return 0.0f;
    float x3 = x * x * x;
    float inner = x + k1 * x3;
    float tanh_arg = clamp(k0 * inner, -10.0f, 10.0f);
    return 0.5f * x * (1.0f + tanh(tanh_arg));
}

static inline float silu(float x, float alpha) {
    return x / (1.0f + exp(-alpha * x));
}

// ============================================================================
// Pass A: FC1 - Compute hidden states with gating + activation
// ============================================================================
// Computes: hidden[m, n] = activation(X[m, k] @ W13_up[expert, n, k] ⊙ gate(X[m, k] @ W13_gate[expert, n, k]))
//
// Tile sizes: BM=16, BN=64, BK=64
// Each threadgroup computes a [BM, BN] tile of output

template<typename T, typename AccumT = float>
void moe_two_pass_prefill_pass_a_impl(
    device const T* X_perm,                // [total_rows, D]
    device const uint* expert_offsets,      // [E + 1]
    device const T* W13_all,                // [E, 2*FF, D]
    device const T* up_biases,              // [E, 2*FF]
    device float* hidden_out,               // [total_rows, FF] - f32 for activation precision
    uint d_model,
    uint d_ff,
    uint E,
    float gate_clip_min,
    float gate_clip_max,
    float up_clip_min,
    float up_clip_max,
    float silu_alpha,
    uint expert_idx,
    uint tile_m,
    uint tile_n,
    ushort simd_gid,
    ushort simd_lid,
    threadgroup T* Xs,
    threadgroup T* Wk_up,
    threadgroup T* Wk_gate
) {
    // Tile configuration - using N-streaming to reduce register pressure
    constexpr short BM = 16;
    constexpr short BN_FULL = 64;         // Full tile size for dispatch
    constexpr short BN_STRIPE = 32;       // Process N in stripes to reduce registers
    constexpr short BK = 64;               // Increased from 48 for better Xs reuse
    constexpr short WM = 2;  // Warps in M
    constexpr short WN = 2;  // Warps in N
    constexpr short tgp_size = WM * WN * 32;

    constexpr short lda_tgp = BK;
    constexpr short ldb_tgp = BN_STRIPE;

    if (expert_idx >= E) return;

    const uint seg_start = expert_offsets[expert_idx];
    const uint seg_end = expert_offsets[expert_idx + 1];
    const uint seg_len = (seg_end > seg_start) ? (seg_end - seg_start) : 0u;
    if (seg_len == 0) return;

    const uint m0 = tile_m * BM;
    const uint n0 = tile_n * BN_FULL;

    if (m0 >= seg_len || n0 >= d_ff) return;

    const uint m_rows = min((uint)BM, seg_len - m0);
    const uint n_cols_full = min((uint)BN_FULL, d_ff - n0);

    // Base addresses
    const ulong x_base = (ulong)(seg_start + m0) * (ulong)d_model;
    const ulong w13_expert_base = (ulong)expert_idx * (ulong)(2 * d_ff) * (ulong)d_model;
    const ulong bias_base = (ulong)expert_idx * (ulong)(2 * d_ff);

    // Common constants
    const uint k_iterations = d_model / BK;
    const uint k_remainder = d_model % BK;
    const ushort thread_idx = simd_gid * 32 + simd_lid;

    // Loop over N dimension in stripes to reduce register pressure
    constexpr uint num_stripes = BN_FULL / BN_STRIPE;
    for (uint stripe = 0; stripe < num_stripes; ++stripe) {
        const uint stripe_n0 = n0 + stripe * BN_STRIPE;
        if (stripe_n0 >= d_ff) break;

        const uint n_cols = min((uint)BN_STRIPE, n_cols_full - stripe * BN_STRIPE);
        if (n_cols == 0) break;

        // Initialize MMA for this stripe
        using mma_t = matmul_utils::BlockMMA<T, AccumT, BM, BN_STRIPE, BK, WM, WN, false, false, lda_tgp, ldb_tgp, AccumT>;
        mma_t mma_up(simd_gid, simd_lid);
        mma_t mma_gate(simd_gid, simd_lid);

        mma_up.Ctile.clear();
        if (GATING_SEL > 1u) {
            mma_gate.Ctile.clear();
        }

        // Main loop over K dimension

        for (uint k_iter = 0; k_iter < k_iterations; ++k_iter) {
            const uint k0 = k_iter * BK;

            // Load X tile [BM, BK]
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (ushort idx = thread_idx; idx < BM * BK; idx += tgp_size) {
                const ushort mi = idx / BK;
                const ushort kk = idx % BK;
                T val = T(0.0f);
                if (mi < m_rows) {
                    val = X_perm[x_base + (ulong)mi * (ulong)d_model + (ulong)(k0 + kk)];
                }
                Xs[mi * BK + kk] = val;
            }

            // Load W13_up tile [BK, BN_STRIPE]
            for (ushort idx = thread_idx; idx < BK * BN_STRIPE; idx += tgp_size) {
                const ushort kk = idx / BN_STRIPE;
                const ushort nn = idx % BN_STRIPE;
                T val = T(0.0f);
                if (nn < n_cols) {
                    const ulong up_idx = w13_expert_base + (ulong)(stripe_n0 + nn) * (ulong)d_model + (ulong)(k0 + kk);
                    val = W13_all[up_idx];
                }
                Wk_up[kk * BN_STRIPE + nn] = val;
            }

            // Load W13_gate tile if needed
            if (GATING_SEL > 1u) {
                for (ushort idx = thread_idx; idx < BK * BN_STRIPE; idx += tgp_size) {
                    const ushort kk = idx / BN_STRIPE;
                    const ushort nn = idx % BN_STRIPE;
                    T val = T(0.0f);
                    if (nn < n_cols) {
                        const ulong gate_idx = w13_expert_base + (ulong)(d_ff + stripe_n0 + nn) * (ulong)d_model + (ulong)(k0 + kk);
                        val = W13_all[gate_idx];
                    }
                    Wk_gate[kk * BN_STRIPE + nn] = val;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // MMA
            mma_up.mma(Xs, Wk_up);
            if (GATING_SEL > 1u) {
                mma_gate.mma(Xs, Wk_gate);
            }
        }

        // Handle K remainder
        if (k_remainder > 0) {
            const uint k0 = k_iterations * BK;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (ushort idx = thread_idx; idx < BM * BK; idx += tgp_size) {
                const ushort mi = idx / BK;
                const ushort kk = idx % BK;
                T val = T(0.0f);
                if (mi < m_rows && kk < k_remainder) {
                    val = X_perm[x_base + (ulong)mi * (ulong)d_model + (ulong)(k0 + kk)];
                }
                Xs[mi * BK + kk] = val;
            }

            for (ushort idx = thread_idx; idx < BK * BN_STRIPE; idx += tgp_size) {
                const ushort kk = idx / BN_STRIPE;
                const ushort nn = idx % BN_STRIPE;
                T val = T(0.0f);
                if (nn < n_cols && kk < k_remainder) {
                    const ulong up_idx = w13_expert_base + (ulong)(stripe_n0 + nn) * (ulong)d_model + (ulong)(k0 + kk);
                    val = W13_all[up_idx];
                }
                Wk_up[kk * BN_STRIPE + nn] = val;
            }

            if (GATING_SEL > 1u) {
                for (ushort idx = thread_idx; idx < BK * BN_STRIPE; idx += tgp_size) {
                    const ushort kk = idx / BN_STRIPE;
                    const ushort nn = idx % BN_STRIPE;
                    T val = T(0.0f);
                    if (nn < n_cols && kk < k_remainder) {
                        const ulong gate_idx = w13_expert_base + (ulong)(d_ff + stripe_n0 + nn) * (ulong)d_model + (ulong)(k0 + kk);
                        val = W13_all[gate_idx];
                    }
                    Wk_gate[kk * BN_STRIPE + nn] = val;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            mma_up.mma(Xs, Wk_up);
            if (GATING_SEL > 1u) {
                mma_gate.mma(Xs, Wk_gate);
            }
        }

        // Store results to hidden buffer with bias and activation
        device float* hidden_tile = hidden_out + (seg_start + m0) * d_ff + stripe_n0;

        const short2 dst_dims = short2(n_cols, m_rows);

        // Custom store with bias + activation
        const short sm = mma_up.sm;
        const short sn = mma_up.sn;

        constexpr short TM = BM / (8 * WM);
        constexpr short TN = BN_STRIPE / (8 * WN);

        UZU_PRAGMA_UNROLL
        for (short tm = 0; tm < TM; ++tm) {
            UZU_PRAGMA_UNROLL
            for (short tn = 0; tn < TN; ++tn) {
                const short m_frag = sm + tm * 8 * WM;
                const short n_frag = sn + tn * 8 * WN;

                if (m_frag >= dst_dims.y || n_frag >= dst_dims.x) continue;

                thread auto& frag_up = mma_up.Ctile.frag_at(tm, tn);

                // Get gate fragment if needed
                float gate_vals[2];
                if (GATING_SEL > 1u) {
                    thread auto& frag_gate = mma_gate.Ctile.frag_at(tm, tn);
                    gate_vals[0] = frag_gate[0];
                    gate_vals[1] = frag_gate[1];
                }

                // Process 2 elements per fragment (8x8 / 32 threads = 2 per thread)
                for (short elem = 0; elem < 2; ++elem) {
                    const short elem_m = m_frag + (elem / 2);
                    const short elem_n = n_frag + (elem % 2);

                    if (elem_m >= dst_dims.y || elem_n >= dst_dims.x) continue;

                    float up_val = frag_up[elem];

                    // Add bias
                    up_val += float(up_biases[bias_base + stripe_n0 + elem_n]);
                    up_val = clamp(up_val, up_clip_min, up_clip_max);

                    float result;
                    if (GATING_SEL <= 1u) {
                        result = (GATING_SEL == 0u) ? gelu_approx(up_val) : silu(up_val, silu_alpha);
                    } else {
                        float gate_val = gate_vals[elem];
                        gate_val += float(up_biases[bias_base + d_ff + stripe_n0 + elem_n]);
                        gate_val = clamp(gate_val, gate_clip_min, gate_clip_max);
                        const float swish_y = (GATING_SEL == 2u) ? silu(gate_val, silu_alpha) : gelu_approx(gate_val);
                        result = swish_y * up_val;
                    }

                    hidden_tile[elem_m * d_ff + elem_n] = result;
                }
            }
        }
    } // end stripe loop
}

// Kernel entry point for Pass A
#define MOE_PASS_A_KERNEL(DTYPE, SUFFIX) \
kernel void moe_two_pass_prefill_pass_a_##SUFFIX( \
    device const DTYPE* X_perm [[buffer(0)]], \
    device const uint* expert_offsets [[buffer(1)]], \
    device const DTYPE* W13_all [[buffer(2)]], \
    device const DTYPE* up_biases [[buffer(3)]], \
    device float* hidden_out [[buffer(4)]], \
    constant uint& d_model [[buffer(5)]], \
    constant uint& d_ff [[buffer(6)]], \
    constant uint& E [[buffer(7)]], \
    constant float& gate_clip_min [[buffer(8)]], \
    constant float& gate_clip_max [[buffer(9)]], \
    constant float& up_clip_min [[buffer(10)]], \
    constant float& up_clip_max [[buffer(11)]], \
    constant float& silu_alpha [[buffer(12)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    ushort simd_gid [[simdgroup_index_in_threadgroup]], \
    ushort simd_lid [[thread_index_in_simdgroup]]) \
{ \
    constexpr short BM = 16; \
    constexpr short BN_STRIPE = 32; \
    constexpr short BK = 64; \
    \
    threadgroup DTYPE Xs_local[BM * BK]; \
    threadgroup DTYPE Wk_up_local[BK * BN_STRIPE]; \
    threadgroup DTYPE Wk_gate_local[BK * BN_STRIPE]; \
    \
    moe_two_pass_prefill_pass_a_impl<DTYPE, float>( \
        X_perm, expert_offsets, W13_all, up_biases, hidden_out, \
        d_model, d_ff, E, \
        gate_clip_min, gate_clip_max, up_clip_min, up_clip_max, silu_alpha, \
        tgpig.z, tgpig.x, tgpig.y, \
        simd_gid, simd_lid, \
        Xs_local, Wk_up_local, Wk_gate_local); \
}

MOE_PASS_A_KERNEL(bfloat, bf16)
MOE_PASS_A_KERNEL(half, f16)
MOE_PASS_A_KERNEL(float, f32)

// Indirect variant (consumes tile map)
template<typename T>
void moe_two_pass_prefill_pass_a_indirect_impl(
    device const T* X_perm,
    device const uint* expert_offsets,
    device const T* W13_all,
    device const T* up_biases,
    device float* hidden_out,
    device const uint* tile_map, // [total_tiles * 3] -> [expert, seg_start, tile_row_offset]
    uint d_model,
    uint d_ff,
    uint E,
    float gate_clip_min,
    float gate_clip_max,
    float up_clip_min,
    float up_clip_max,
    float silu_alpha,
    uint row_tile_idx,
    uint n_tile_idx,
    ushort simd_gid,
    ushort simd_lid,
    threadgroup T* Xs,
    threadgroup T* Wk_up,
    threadgroup T* Wk_gate) {
    constexpr uint ROW_TILE = 16;
    const uint base = row_tile_idx * 3u;
    const uint expert_idx = tile_map[base + 0u];
    if (expert_idx >= E) {
        return;
    }
    const uint tile_m0 = tile_map[base + 2u];
    const uint tile_m = tile_m0 / ROW_TILE;

    moe_two_pass_prefill_pass_a_impl<T, float>(
        X_perm,
        expert_offsets,
        W13_all,
        up_biases,
        hidden_out,
        d_model,
        d_ff,
        E,
        gate_clip_min,
        gate_clip_max,
        up_clip_min,
        up_clip_max,
        silu_alpha,
        expert_idx,
        tile_m,
        n_tile_idx,
        simd_gid,
        simd_lid,
        Xs,
        Wk_up,
        Wk_gate);
}

#define MOE_PASS_A_INDIRECT_KERNEL(DTYPE, SUFFIX) \
kernel void moe_two_pass_prefill_pass_a_indirect_##SUFFIX( \
    device const DTYPE* X_perm [[buffer(0)]], \
    device const uint* expert_offsets [[buffer(1)]], \
    device const DTYPE* W13_all [[buffer(2)]], \
    device const DTYPE* up_biases [[buffer(3)]], \
    device float* hidden_out [[buffer(4)]], \
    constant uint& d_model [[buffer(5)]], \
    constant uint& d_ff [[buffer(6)]], \
    constant uint& E [[buffer(7)]], \
    constant float& gate_clip_min [[buffer(8)]], \
    constant float& gate_clip_max [[buffer(9)]], \
    constant float& up_clip_min [[buffer(10)]], \
    constant float& up_clip_max [[buffer(11)]], \
    constant float& silu_alpha [[buffer(12)]], \
    device const uint* tile_map [[buffer(13)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    ushort simd_gid [[simdgroup_index_in_threadgroup]], \
    ushort simd_lid [[thread_index_in_simdgroup]]) \
{ \
    threadgroup DTYPE Xs_local[16 * 64]; \
    threadgroup DTYPE Wk_up_local[64 * 32]; \
    threadgroup DTYPE Wk_gate_local[64 * 32]; \
    \
    moe_two_pass_prefill_pass_a_indirect_impl<DTYPE>( \
        X_perm, expert_offsets, W13_all, up_biases, hidden_out, tile_map, \
        d_model, d_ff, E, gate_clip_min, gate_clip_max, up_clip_min, up_clip_max, \
        silu_alpha, tgpig.y, tgpig.x, simd_gid, simd_lid, \
        Xs_local, Wk_up_local, Wk_gate_local); \
}

MOE_PASS_A_INDIRECT_KERNEL(bfloat, bf16)
MOE_PASS_A_INDIRECT_KERNEL(half, f16)
MOE_PASS_A_INDIRECT_KERNEL(float, f32)

// ============================================================================
// Pass B: FC2 - Compute output from hidden states
// ============================================================================
// Computes: output[m, n] = hidden[m, k] @ W2[expert, k, n] + bias[n]
//
// Tile sizes: BM=16, BN=64, BK=64
// Each threadgroup computes a [BM, BN] tile of output

template<typename T, typename AccumT = float>
void moe_two_pass_prefill_pass_b_impl(
    device const float* hidden,             // [total_rows, FF] - f32 from Pass A
    device const uint* expert_offsets,      // [E + 1]
    device const T* W2_all,                 // [E, FF, D]
    device const T* down_biases,            // [E, D]
    device T* output,                       // [total_rows, D]
    uint d_model,
    uint d_ff,
    uint E,
    uint expert_idx,
    uint tile_m,
    uint tile_n,
    ushort simd_gid,
    ushort simd_lid,
    threadgroup T* Hs,
    threadgroup T* Wk
) {
    // Tile configuration
    constexpr short BM = 16;
    constexpr short BN = 64;
    constexpr short BK = 48;
    constexpr short WM = 2;  // Warps in M
    constexpr short WN = 2;  // Warps in N
    constexpr short tgp_size = WM * WN * 32;

    constexpr short lda_tgp = BK;
    constexpr short ldb_tgp = BN;

    if (expert_idx >= E) return;

    const uint seg_start = expert_offsets[expert_idx];
    const uint seg_end = expert_offsets[expert_idx + 1];
    const uint seg_len = (seg_end > seg_start) ? (seg_end - seg_start) : 0u;
    if (seg_len == 0) return;

    const uint m0 = tile_m * BM;
    const uint n0 = tile_n * BN;

    if (m0 >= seg_len || n0 >= d_model) return;

    const uint m_rows = min((uint)BM, seg_len - m0);
    const uint n_cols = min((uint)BN, d_model - n0);

    // Base addresses
    const ulong h_base = (ulong)(seg_start + m0) * (ulong)d_ff;
    const ulong w2_expert_base = (ulong)expert_idx * (ulong)d_ff * (ulong)d_model;
    const ulong bias_base = (ulong)expert_idx * (ulong)d_model;

    // Initialize MMA
    using mma_t = matmul_utils::BlockMMA<T, AccumT, BM, BN, BK, WM, WN, false, false, lda_tgp, ldb_tgp, AccumT>;
    mma_t mma(simd_gid, simd_lid);
    mma.Ctile.clear();

    // Main loop over K dimension (FF)
    const uint k_iterations = d_ff / BK;
    const uint k_remainder = d_ff % BK;

    for (uint k_iter = 0; k_iter < k_iterations; ++k_iter) {
        const uint k0 = k_iter * BK;

        // Load hidden tile [BM, BK]
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ushort thread_idx = simd_gid * 32 + simd_lid;
        for (ushort idx = thread_idx; idx < BM * BK; idx += tgp_size) {
            const ushort mi = idx / BK;
            const ushort kk = idx % BK;
            T val = T(0.0f);
            if (mi < m_rows) {
                val = T(hidden[h_base + (ulong)mi * (ulong)d_ff + (ulong)(k0 + kk)]);
            }
            Hs[mi * BK + kk] = val;
        }

        // Load W2 tile [BK, BN] - W2 is [E, FF, D], D is contiguous
        for (ushort idx = thread_idx; idx < BK * BN; idx += tgp_size) {
            const ushort kk = idx / BN;
            const ushort nn = idx % BN;
            T val = T(0.0f);
            if (nn < n_cols) {
                // W2 layout: [E, D, FF] - FF contiguous
                const ulong w2_idx =
                    w2_expert_base + (ulong)(n0 + nn) * (ulong)d_ff + (ulong)(k0 + kk);
                val = W2_all[w2_idx];
            }
            Wk[kk * BN + nn] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA
        mma.mma(Hs, Wk);
    }

    // Handle K remainder
    if (k_remainder > 0) {
        const uint k0 = k_iterations * BK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ushort thread_idx = simd_gid * 32 + simd_lid;
        for (ushort idx = thread_idx; idx < BM * BK; idx += tgp_size) {
            const ushort mi = idx / BK;
            const ushort kk = idx % BK;
            T val = T(0.0f);
            if (mi < m_rows && kk < k_remainder) {
                val = T(hidden[h_base + (ulong)mi * (ulong)d_ff + (ulong)(k0 + kk)]);
            }
            Hs[mi * BK + kk] = val;
        }

        for (ushort idx = thread_idx; idx < BK * BN; idx += tgp_size) {
            const ushort kk = idx / BN;
            const ushort nn = idx % BN;
            T val = T(0.0f);
            if (nn < n_cols && kk < k_remainder) {
                const ulong w2_idx =
                    w2_expert_base + (ulong)(n0 + nn) * (ulong)d_ff + (ulong)(k0 + kk);
                val = W2_all[w2_idx];
            }
            Wk[kk * BN + nn] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma.mma(Hs, Wk);
    }

    // Store results with bias
    device T* output_tile = output + (seg_start + m0) * d_model + n0;

    // Add bias before storing
    const short sm = mma.sm;
    const short sn = mma.sn;

    constexpr short TM = BM / (8 * WM);
    constexpr short TN = BN / (8 * WN);

    const short2 dst_dims = short2(n_cols, m_rows);

    UZU_PRAGMA_UNROLL
    for (short tm = 0; tm < TM; ++tm) {
        UZU_PRAGMA_UNROLL
        for (short tn = 0; tn < TN; ++tn) {
            const short m_frag = sm + tm * 8 * WM;
            const short n_frag = sn + tn * 8 * WN;

            if (m_frag >= dst_dims.y || n_frag >= dst_dims.x) continue;

            thread auto& frag = mma.Ctile.frag_at(tm, tn);

            // Process 2 elements per fragment
            for (short elem = 0; elem < 2; ++elem) {
                const short elem_m = m_frag + (elem / 2);
                const short elem_n = n_frag + (elem % 2);

                if (elem_m >= dst_dims.y || elem_n >= dst_dims.x) continue;

                float val = frag[elem];
                val += float(down_biases[bias_base + n0 + elem_n]);

                output_tile[elem_m * d_model + elem_n] = T(val);
            }
        }
    }
}

// Kernel entry point for Pass B
#define MOE_PASS_B_KERNEL(DTYPE, SUFFIX) \
kernel void moe_two_pass_prefill_pass_b_##SUFFIX( \
    device const float* hidden [[buffer(0)]], \
    device const uint* expert_offsets [[buffer(1)]], \
    device const DTYPE* W2_all [[buffer(2)]], \
    device const DTYPE* down_biases [[buffer(3)]], \
    device DTYPE* output [[buffer(4)]], \
    constant uint& d_model [[buffer(5)]], \
    constant uint& d_ff [[buffer(6)]], \
    constant uint& E [[buffer(7)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    ushort simd_gid [[simdgroup_index_in_threadgroup]], \
    ushort simd_lid [[thread_index_in_simdgroup]]) \
{ \
    constexpr short BM = 16; \
    constexpr short BN = 64; \
    constexpr short BK = 48; \
    \
    threadgroup DTYPE Hs_local[BM * BK]; \
    threadgroup DTYPE Wk_local[BK * BN]; \
    \
    moe_two_pass_prefill_pass_b_impl<DTYPE, float>( \
        hidden, expert_offsets, W2_all, down_biases, output, \
        d_model, d_ff, E, \
        tgpig.z, tgpig.x, tgpig.y, \
        simd_gid, simd_lid, \
        Hs_local, Wk_local); \
}

MOE_PASS_B_KERNEL(bfloat, bf16)
MOE_PASS_B_KERNEL(half, f16)
MOE_PASS_B_KERNEL(float, f32)

template<typename T>
void moe_two_pass_prefill_pass_b_indirect_impl(
    device const float* hidden,
    device const uint* expert_offsets,
    device const T* W2_all,
    device const T* down_biases,
    device T* output,
    device const uint* tile_map,
    uint d_model,
    uint d_ff,
    uint E,
    uint row_tile_idx,
    uint n_tile_idx,
    ushort simd_gid,
    ushort simd_lid,
    threadgroup T* Hs,
    threadgroup T* Wk) {
    constexpr uint ROW_TILE = 16;
    const uint base = row_tile_idx * 3u;
    const uint expert_idx = tile_map[base + 0u];
    if (expert_idx >= E) {
        return;
    }
    const uint tile_m0 = tile_map[base + 2u];
    const uint tile_m = tile_m0 / ROW_TILE;

    moe_two_pass_prefill_pass_b_impl<T, float>(
        hidden,
        expert_offsets,
        W2_all,
        down_biases,
        output,
        d_model,
        d_ff,
        E,
        expert_idx,
        tile_m,
        n_tile_idx,
        simd_gid,
        simd_lid,
        Hs,
        Wk);
}

#define MOE_PASS_B_INDIRECT_KERNEL(DTYPE, SUFFIX) \
kernel void moe_two_pass_prefill_pass_b_indirect_##SUFFIX( \
    device const float* hidden [[buffer(0)]], \
    device const uint* expert_offsets [[buffer(1)]], \
    device const DTYPE* W2_all [[buffer(2)]], \
    device const DTYPE* down_biases [[buffer(3)]], \
    device DTYPE* output [[buffer(4)]], \
    constant uint& d_model [[buffer(5)]], \
    constant uint& d_ff [[buffer(6)]], \
    constant uint& E [[buffer(7)]], \
    device const uint* tile_map [[buffer(8)]], \
    uint3 tgpig [[threadgroup_position_in_grid]], \
    ushort simd_gid [[simdgroup_index_in_threadgroup]], \
    ushort simd_lid [[thread_index_in_simdgroup]]) \
{ \
    threadgroup DTYPE Hs_local[16 * 48]; \
    threadgroup DTYPE Wk_local[48 * 64]; \
    \
    moe_two_pass_prefill_pass_b_indirect_impl<DTYPE>( \
        hidden, expert_offsets, W2_all, down_biases, output, tile_map, \
        d_model, d_ff, E, tgpig.y, tgpig.x, simd_gid, simd_lid, \
        Hs_local, Wk_local); \
}

MOE_PASS_B_INDIRECT_KERNEL(bfloat, bf16)
MOE_PASS_B_INDIRECT_KERNEL(half, f16)
MOE_PASS_B_INDIRECT_KERNEL(float, f32)
